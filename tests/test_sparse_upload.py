"""Run with python -m unittest discover -s tests -v from the core directory."""

import io
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fedckd.client import FedCKDClient
from fedckd import server as server_module
from fedckd.server import FedCKDServer, construct_teacher_labels


torch.set_num_threads(1)


def make_client(mask, dtype=torch.float32, client_id=0):
    model = nn.Linear(2, 3)
    model.num_classes = 3
    client = FedCKDClient(model, [(torch.zeros(2), 0)], client_id, 42, 'cpu')
    client.public_masks = torch.tensor(mask, dtype=torch.float32)
    logits = torch.arange(len(mask) * 3, dtype=dtype).reshape(len(mask), 3) / 7
    logits[~client.public_masks.bool()] = -1
    client.public_logits = logits
    client.prototype_head_audit = {'mismatch_fraction': 0.5}
    return client


class SparseUploadTests(unittest.TestCase):
    def unpack(self, package):
        self.assertTrue(hasattr(server_module, 'unpack_upload_package'),
                        'server must reconstruct sparse uploads at the receive boundary')
        return server_module.unpack_upload_package(package)

    def test_wire_package_contains_only_active_logits_and_packed_mask(self):
        client = make_client([1, 0, 1, 0, 0, 1, 0, 0, 1])
        package = client.prepare_upload_package()
        self.assertIn('sparse_logits', package)
        self.assertNotIn('public_logits', package)
        self.assertNotIn('mask_vector', package)
        self.assertEqual(package['sparse_logits'].shape, (4, 3))
        self.assertEqual(package['packed_mask'].dtype, np.uint8)
        self.assertEqual(package['packed_mask'].nbytes, 2)
        self.assertEqual(package['num_pub'], 9)
        self.assertEqual(package['num_classes'], 3)
        self.assertEqual(package['train_samples'], client.train_samples)
        np.testing.assert_array_equal(package['label_type'], client.label_type)
        np.testing.assert_array_equal(package['label_cnt'], client.label_cnt)
        self.assertEqual(package['prototype_head_audit'], client.prototype_head_audit)
        # Transport data must not share the original dense tensor storage.
        self.assertNotEqual(package['sparse_logits'].untyped_storage().data_ptr(),
                            client.public_logits.untyped_storage().data_ptr())

    def test_exact_round_trip_including_empty_full_and_padding_bits(self):
        for mask in ([], [0] * 9, [1] * 9, [1, 0, 1, 0, 1, 0, 0, 1, 1]):
            for dtype in (torch.float16, torch.float32, torch.float64):
                with self.subTest(mask=mask, dtype=dtype):
                    client = make_client(mask, dtype)
                    package = client.prepare_upload_package()
                    # Simulate an actual serialization boundary.
                    stream = io.BytesIO()
                    torch.save(package, stream)
                    stream.seek(0)
                    received = torch.load(stream, weights_only=False)
                    logits, restored_mask = self.unpack(received)
                    self.assertTrue(torch.equal(logits, client.public_logits))
                    self.assertTrue(torch.equal(restored_mask, client.public_masks.bool()))
                    self.assertEqual(logits.dtype, dtype)
                    self.assertEqual(logits.device, package['sparse_logits'].device)
                    self.assertEqual(restored_mask.dtype, torch.bool)

    def test_optional_alignment_parameters_are_preserved(self):
        client = make_client([0, 1])
        self.assertNotIn('local_model_params', client.prepare_upload_package())
        package = client.prepare_upload_package(include_model_params=True)
        for key, value in client.model.state_dict().items():
            self.assertTrue(torch.equal(value, package['local_model_params'][key]))
        self.assertTrue(torch.equal(self.unpack(package)[0], client.public_logits))

    def test_5000_sample_mask_is_625_bytes(self):
        client = make_client([1, 0] * 2500)
        package = client.prepare_upload_package()
        self.assertIn('packed_mask', package)
        self.assertEqual(package['packed_mask'].nbytes, 625)
        self.assertEqual(package['sparse_logits'].numel(), 2500 * 3)

    def test_malformed_payload_is_rejected(self):
        package = make_client([1, 0, 1]).prepare_upload_package()
        for change in ({'packed_mask': np.array([], dtype=np.uint8)},
                       {'sparse_logits': torch.zeros(1, 3)},
                       {'num_classes': 4}, {'num_pub': -1}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.unpack(dict(package, **change))

    def test_client_rejects_nonbinary_or_misaligned_mask(self):
        for mask in (torch.tensor([1., 0.5]), torch.ones(3)):
            client = make_client([1, 0])
            client.public_masks = mask
            with self.subTest(mask=mask), self.assertRaises(ValueError):
                client.prepare_upload_package()

    def test_teacher_and_real_server_updates_match_dense_input(self):
        masks = [[1] * 9, [1, 0, 1, 1, 0, 1, 1, 1, 1],
                 [1, 0, 1, 1, 1, 1, 1, 1, 1], [1] * 9]
        loader = DataLoader(TensorDataset(torch.arange(18).reshape(9, 2).float() / 18,
                                         torch.zeros(9)), batch_size=4)
        for round_id in (0, 4, 5, 100):
            for scenario in ('teacher', 'no_teacher', 'no_eligible'):
                with self.subTest(round_id=round_id, scenario=scenario):
                    clients = [make_client(mask, client_id=i) for i, mask in enumerate(masks)]
                    clients[-1].prototype_head_audit['mismatch_fraction'] = 0.75
                    if scenario == 'no_teacher':
                        for client in clients:
                            client.public_masks.zero_()
                            client.public_logits.fill_(-1)
                    if scenario == 'no_eligible':
                        for client in clients:
                            client.prototype_head_audit['mismatch_fraction'] = 1.0
                    packages = [c.prepare_upload_package(round_id < 5) for c in clients]
                    dense = [(c.public_logits, c.public_masks.bool()) for c in clients]
                    restored = [self.unpack(p) for p in packages]
                    audits = [c.prototype_head_audit for c in clients]
                    expected_teacher = construct_teacher_labels(
                        torch.stack([x[0] for x in dense]),
                        torch.stack([x[1] for x in dense]), audits)
                    actual_teacher = construct_teacher_labels(
                        torch.stack([x[0] for x in restored]),
                        torch.stack([x[1] for x in restored]), audits)
                    self.assertTrue(torch.equal(expected_teacher[0], actual_teacher[0]))
                    self.assertTrue(torch.equal(expected_teacher[1], actual_teacher[1]))
                    self.assertEqual(expected_teacher[2], actual_teacher[2])
                    reference = FedCKDServer(clients[0].model, 'cpu')
                    actual = FedCKDServer(clients[0].model, 'cpu')
                    torch.manual_seed(42)
                    with patch.object(server_module, 'unpack_upload_package', side_effect=dense):
                        reference.update(packages, loader, round_id)
                    torch.manual_seed(42)
                    actual.update(packages, loader, round_id)
                    for key, value in reference.global_model.state_dict().items():
                        self.assertTrue(torch.equal(value, actual.global_model.state_dict()[key]), key)


if __name__ == '__main__':
    unittest.main()
