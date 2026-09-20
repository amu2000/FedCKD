# FedCKD Core

Core algorithm components extracted from the FedCKD implementation. This edition
is intended for reading and integration. It does not include an executable
experiment, command-line entry point, dataset preparation, partition manifests,
attack simulation, logging, evaluation, checkpoints, or experiment results.

## Components

| File | Role |
| --- | --- |
| `fedckd/client.py` | Local SGD, private class prototypes, percentile radius, public masks, prototype-head checks, upload package |
| `fedckd/server.py` | Client eligibility, majority-class logits, five-round task alignment, KL distillation |
| `fedckd/pretraining.py` | Two-view image augmentation, SimCLR projection/loss, feature pretraining, classification-head reset |
| `fedckd/model.py` | Fashion-MNIST CNN backbone with feature output |

## Algorithm conventions

The prototype distance percentile is 0.8. The public mask compares the nearest
prototype cosine distance against the mean of the available class radii.
Prototypes stay on the client; the semantic check uploads scalar statistics.
A mismatch fraction strictly greater than 0.5 rejects a client. Each public
sample needs at least three eligible masked voters before majority-class
selection; consensus logits are averaged. Ties follow client order.

Local training uses two SGD epochs, batch size 16, learning rate 0.01,
momentum 0.9, and weight decay 0.0005. The optimizer resets each round.
The source's round-local scheduler did not change the learning rate used for
training; this component therefore keeps the same constant rate without that
ineffective scheduler.

Rounds 0-4 initialize the student from eligible clients, weighted by their
private sample counts. This task alignment also occurs when eligible clients
exist but no public sample has a valid teacher. After round 4 there is no
client-parameter aggregation. No eligible clients always retains the model;
no valid teacher after round 4 also retains the model. Rejected clients are
never used as a fallback. Distillation uses three epochs, SGD with momentum
0.9, learning rate 0.001, batch size 512, temperature 1.0, and cosine annealing.

## Component interfaces

The modules use PyTorch, torchvision, and NumPy. They expose components rather
than a complete training program. A caller supplies the model, device, data,
and round scheduling. The reference setting uses 16 of 20 clients each round
and 5,000 unlabeled public EMNIST images; neither sampling nor data selection
is implemented here.

`FashionCNN(1, 10)` exposes `fc` and `forward(..., return_hidden=True)`.
`pretrain_fashion_cnn_simclr` accepts a two-view dataset; the reference settings
are 20 epochs, batch size 256, learning rate 0.001, temperature 0.5, and projection
dimension 128. `TwoViewDataset` ignores the dataset's labels.

Clients receive private `(image, label)` pairs. After `set_parameters` and
`train(round_id)`, `process_public_data_and_upload` builds the upload package;
`include_model_params` is true only for rounds 0-4. Public loaders yield
`(image, unused_label)` pairs in the same fixed sample order on every client
and the server, without shuffling or stochastic transforms. Public labels
are never used. Upload tensors are on the CPU and package order determines
the voting tie order. The server's `update` consumes these packages and the
same public loader.

This extraction is not a full reproduction package. Full-run accuracy results
from the runnable edition are not asserted for this component-only edition.

## License

Apache License 2.0. See `LICENSE` and `THIRD_PARTY_NOTICES.md`.
