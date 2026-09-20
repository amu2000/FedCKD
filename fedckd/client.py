# Adapted from TFLlib (Apache-2.0); extracted and simplified for FedCKD.
# See THIRD_PARTY_NOTICES.md.
"""Private training, prototype masks, and client-local semantic checks."""

import copy
import hashlib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


class FedCKDClient:
    """A client component; no dataset download, attack simulation, or logging.

    ``train_data`` yields private (image, label) pairs. Public loaders must be
    ordered identically on all clients and the server; public labels are ignored.
    """

    def __init__(self, model, train_data, client_id, seed, device):
        self.model = copy.deepcopy(model)
        self.train_data = train_data
        self.id = client_id
        self.seed = seed
        self.device = torch.device(device)
        self.num_classes = model.num_classes
        self.train_samples = len(train_data)
        labels = [label for _, label in train_data]
        self.label_type, self.label_cnt = np.unique(labels, return_counts=True)
        self.prototype_percentile = 0.8
        self.prototype_head_audit = None

    def set_parameters(self, model):
        self.model.load_state_dict(
            {key: value.clone().detach() for key, value in model.state_dict().items()}
        )

    def train(self, round_id):
        """Two local SGD epochs at lr=0.01; the optimizer resets each round."""
        seed_key = f"{int(self.seed)}:{int(self.id)}:{int(round_id)}"
        digest = hashlib.blake2b(seed_key.encode("ascii"), digest_size=8).digest()
        generator_seed = int.from_bytes(digest, byteorder="big") & (1 << 63) - 1
        generator = torch.Generator().manual_seed(generator_seed)
        self.trainloader = DataLoader(
            self.train_data,
            batch_size=16,
            drop_last=False,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            generator=generator,
        )
        self.model.train().to(self.device)
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005
        )
        for _ in range(2):
            for images, labels in self.trainloader:
                optimizer.zero_grad()
                if labels.dim() == 2:
                    labels = labels.squeeze(1)
                loss = F.cross_entropy(
                    self.model(images.to(self.device)), labels.to(self.device).long()
                )
                loss.backward()
                optimizer.step()
        self.model.cpu()

    def build_prototypes_and_threshold(self):
        """Build private class means and the mean p=0.8 cosine-distance radius."""
        self.model.eval()
        self.model.to(self.device)
        class_features = {c: [] for c in range(self.num_classes)}
        with torch.no_grad():
            for batch in self.trainloader:
                x, y = batch
                x, y = (x.to(self.device), y.to(self.device))
                _, features = self.model(x, return_hidden=True)
                if y.dim() == 2:
                    y = y.squeeze(1)
                for feat, label in zip(features, y):
                    label_idx = label.item()
                    if label_idx in class_features:
                        class_features[label_idx].append(feat.cpu())
        prototypes = {}
        class_radii = {}
        for c in range(self.num_classes):
            if len(class_features[c]) > 0:
                features_tensor = torch.stack(class_features[c])
                prototypes[c] = features_tensor.mean(dim=0)
                cos_sim = F.cosine_similarity(
                    features_tensor,
                    prototypes[c].unsqueeze(0).expand(features_tensor.size(0), -1),
                    dim=1,
                )
                cos_distances = 1 - cos_sim
                class_radii[c] = torch.quantile(
                    cos_distances, self.prototype_percentile
                ).item()
            else:
                prototypes[c] = None
                class_radii[c] = None
        valid_radii = [r for r in class_radii.values() if r is not None]
        if len(valid_radii) > 0:
            threshold = np.mean(valid_radii)
        else:
            threshold = 1.0
        self._build_prototype_head_audit(prototypes)
        self.model.cpu()
        self.prototypes = prototypes
        self.threshold = threshold
        self.class_radii = class_radii
        return (prototypes, threshold)

    def _build_prototype_head_audit(self, prototypes):
        """Check private raw-label prototypes with the local head; upload only scalars."""
        self.prototype_head_audit = None
        head = getattr(self.model, "fc", None)
        if not callable(head):
            raise RuntimeError(
                "Prototype-head consistency requires a model with an fc head"
            )
        valid_classes = [
            class_id
            for class_id, prototype in prototypes.items()
            if prototype is not None
        ]
        if not valid_classes:
            raise RuntimeError(
                "Prototype-head audit requires at least one private class"
            )
        prototype_batch = torch.stack(
            [prototypes[class_id] for class_id in valid_classes]
        ).to(self.device)
        raw_classes = torch.tensor(valid_classes, dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits = head(prototype_batch)
            probabilities = F.softmax(logits, dim=1)
            predictions = logits.argmax(dim=1)
            match = predictions == raw_classes
            true_logits = logits.gather(1, raw_classes.unsqueeze(1)).squeeze(1)
            competing_logits = logits.clone()
            competing_logits.scatter_(1, raw_classes.unsqueeze(1), float("-inf"))
            true_margins = true_logits - competing_logits.max(dim=1).values
        self.prototype_head_audit = {
            "class_count": int(len(valid_classes)),
            "match_fraction": float(match.float().mean().item()),
            "mismatch_fraction": float((~match).float().mean().item()),
            "mean_true_probability": float(
                probabilities.gather(1, raw_classes.unsqueeze(1)).mean().item()
            ),
            "mean_true_margin": float(true_margins.mean().item()),
        }

    def validate_public_data(self, public_dataloader):
        """Return public masks, masked logits, and nearest-prototype distances.

        Public labels are ignored; rejected sample logits are set to -1."""
        if not hasattr(self, "prototypes") or self.prototypes is None:
            raise RuntimeError(
                "Build private prototypes before validating public images"
            )
        self.model.eval()
        self.model.to(self.device)
        prototypes_gpu = {}
        for c, proto in self.prototypes.items():
            if proto is not None:
                prototypes_gpu[c] = proto.to(self.device)
        all_masks = []
        all_logits = []
        all_min_distances = []
        with torch.no_grad():
            for batch in public_dataloader:
                x, _ = batch
                x = x.to(self.device)
                logits, features = self.model(x, return_hidden=True)
                batch_size = features.size(0)
                min_distances = torch.full(
                    (batch_size,), float("inf"), device=self.device
                )
                for c, proto in prototypes_gpu.items():
                    cos_sim = F.cosine_similarity(
                        features, proto.unsqueeze(0).expand(batch_size, -1), dim=1
                    )
                    cos_dist = 1 - cos_sim
                    min_distances = torch.min(min_distances, cos_dist)
                masks = (min_distances <= self.threshold).float()
                processed_logits = logits.clone()
                ood_mask = masks == 0
                processed_logits[ood_mask] = -1
                all_masks.append(masks.cpu())
                all_logits.append(processed_logits.cpu())
                all_min_distances.append(min_distances.cpu())
        masks = torch.cat(all_masks, dim=0)
        logits = torch.cat(all_logits, dim=0)
        min_distances = torch.cat(all_min_distances, dim=0)
        self.model.cpu()
        return (masks, logits, min_distances)

    def prepare_upload_package(self, include_model_params=False):
        """Package public predictions and audit scalars, plus optional alignment parameters."""
        if not hasattr(self, "public_logits"):
            raise RuntimeError("Process public images before preparing an upload")
        package = {
            "client_id": self.id,
            "public_logits": self.public_logits,
            "mask_vector": self.public_masks,
            "label_type": self.label_type,
            "label_cnt": self.label_cnt,
            "train_samples": self.train_samples,
        }
        if self.prototype_head_audit is not None:
            package["prototype_head_audit"] = dict(self.prototype_head_audit)
        if include_model_params:
            package["local_model_params"] = {
                k: v.clone().detach().cpu() for k, v in self.model.state_dict().items()
            }
        return package

    def process_public_data_and_upload(
        self, public_dataloader, include_model_params=False
    ):
        """Build private prototypes, mask public predictions, and prepare an upload."""
        self.build_prototypes_and_threshold()
        masks, logits, min_distances = self.validate_public_data(public_dataloader)
        self.public_masks = masks
        self.public_logits = logits
        self.public_min_distances = min_distances
        package = self.prepare_upload_package(include_model_params)
        return package
