# Adapted from TFLlib (Apache-2.0); extracted and simplified for FedCKD.
# See THIRD_PARTY_NOTICES.md.
"""Prototype-head eligibility, consensus logits, task alignment, and distillation."""

import copy
from collections import Counter
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


def construct_teacher_labels(client_logits, client_masks, audits):
    """Return teacher logits, a sample mask, and eligible client positions.

    Tensors are CPU tensors with shapes [clients, samples, classes] and
    [clients, samples]. Audit order must match client order. Only a mismatch
    fraction strictly greater than 0.5 rejects a client. At least three eligible
    masked voters are required before taking the most frequent predicted class.
    Ties follow the first encountered class in client order, as in the source.
    """
    count, samples, classes = client_logits.shape
    if client_masks.shape != (count, samples) or len(audits) != count:
        raise ValueError(
            "Logits, masks, and audits must use the same client/sample order"
        )
    mismatches = np.asarray(
        [audit["mismatch_fraction"] for audit in audits], dtype=np.float64
    )
    if not np.isfinite(mismatches).all() or ((mismatches < 0) | (mismatches > 1)).any():
        raise ValueError("Mismatch fractions must be finite and in [0, 1]")
    eligible_mask = torch.from_numpy(mismatches <= 0.5)
    eligible = torch.where(eligible_mask)[0].tolist()
    teacher = torch.zeros(samples, classes)
    valid = torch.zeros(samples, dtype=torch.bool)
    for index in range(samples):
        available = torch.where((client_masks[:, index] == 1) & eligible_mask)[0]
        if len(available) < 3:
            continue
        predictions = client_logits[available, index, :].argmax(dim=1)
        consensus = Counter(predictions.tolist()).most_common(1)[0][0]
        clean = available[torch.where(predictions == consensus)[0]]
        teacher[index] = client_logits[clean, index, :].mean(dim=0)
        valid[index] = True
    return (teacher, valid, eligible)


class FedCKDServer:
    """One-round server update; client scheduling and evaluation are external."""

    def __init__(self, model, device):
        self.global_model = copy.deepcopy(model).cpu()
        self.device = torch.device(device)
        self.distillation_lr = 0.001
        self.distillation_temperature = 1.0
        self.distillation_epochs = 3
        self.distillation_batch_size = 512
        self.distillation_cosine_annealing = True

    def update(self, packages, public_dataloader, round_id):
        """Consume ordered client packages and update the global model.

        Rounds are zero-based. Only rounds 0-4 require local_model_params.
        No eligible clients retains the model. During rounds 0-4, eligible
        clients still perform task alignment if no valid teacher exists;
        after that, no valid teacher retains the previous global model.
        """
        if round_id < 0:
            raise ValueError("round_id must be non-negative")
        if not packages:
            return self.global_model
        self._current_round = round_id
        self.public_dataloader = public_dataloader
        teacher, valid, self._eligible_client_indices = construct_teacher_labels(
            torch.stack([p["public_logits"] for p in packages]),
            torch.stack([p["mask_vector"] for p in packages]),
            [p["prototype_head_audit"] for p in packages],
        )
        self.uploaded_models = []
        self.uploaded_weights = []
        if round_id < 5 and self._eligible_client_indices:
            sample_counts = [p["train_samples"] for p in packages]
            if any((count <= 0 for count in sample_counts)):
                raise ValueError("Each participating client must have private samples")
            total_samples = sum(sample_counts)
            self.uploaded_weights = [count / total_samples for count in sample_counts]
            for package in packages:
                model = copy.deepcopy(self.global_model)
                model.load_state_dict(package["local_model_params"])
                self.uploaded_models.append(model)
        if valid.any():
            self.distill_server_model(teacher, valid)
        elif round_id < 5:
            self._initialize_student()
        return self.global_model

    def _initialize_student(self):
        """Sample-weighted alignment using eligible clients only."""
        eligible = self._eligible_client_indices
        if not eligible:
            return False
        self.global_model = copy.deepcopy(self.uploaded_models[eligible[0]])
        for parameter in self.global_model.parameters():
            parameter.data.zero_()
        weights = [self.uploaded_weights[index] for index in eligible]
        weight_sum = float(sum(weights))
        weights = [weight / weight_sum for weight in weights]
        for weight, index in zip(weights, eligible):
            update = self.uploaded_models[index].state_dict()
            for key, parameter in self.global_model.named_parameters():
                parameter.data += update[key].data.clone() * weight
        return True

    def distill_server_model(self, teacher_logits, valid_mask):
        """Train with consensus soft targets, following eligible-client alignment in rounds 0-4."""
        device = self.device
        # Keep the source's initial iterator creation and resulting RNG advance.
        sample_batch = next(iter(self.public_dataloader))
        _ = len(sample_batch)
        public_images = []
        for batch in self.public_dataloader:
            x, _ = batch
            public_images.append(x)
        public_images = torch.cat(public_images, dim=0)
        valid_indices = torch.where(valid_mask)[0]
        valid_images = public_images[valid_indices]
        valid_teacher_logits = teacher_logits[valid_indices]
        distill_dataset = TensorDataset(valid_images, valid_teacher_logits)
        distill_loader = DataLoader(
            distill_dataset,
            batch_size=self.distillation_batch_size,
            shuffle=True,
            drop_last=False,
        )
        warmup_rounds = 5
        if self._current_round < warmup_rounds:
            self._initialize_student()
        self.global_model.to(device)
        self.global_model.train()
        optimizer = torch.optim.SGD(
            self.global_model.parameters(), lr=self.distillation_lr, momentum=0.9
        )
        scheduler = None
        if self.distillation_cosine_annealing:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.distillation_epochs,
                eta_min=self.distillation_lr * 0.01,
            )
        kl_loss = nn.KLDivLoss(reduction="batchmean")
        T = self.distillation_temperature
        for epoch in range(self.distillation_epochs):
            for batch_data in distill_loader:
                images, teacher_logits_batch = batch_data
                images = images.to(device)
                teacher_logits_batch = teacher_logits_batch.to(device)
                out = self.global_model(images)
                student_logits = out
                teacher_probs = F.softmax(teacher_logits_batch / T, dim=1)
                student_log_probs = F.log_softmax(student_logits / T, dim=1)
                loss = kl_loss(student_log_probs, teacher_probs) * T**2
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
        self.global_model.cpu()
