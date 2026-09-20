# Third-party code

## TFLlib

Original framework authors: Jiahao Chen and the TFLlib contributors.
Source: https://github.com/xaddwell/TFLlib
License: Apache License 2.0, reproduced in `LICENSE`.
Source and license previously inspected at commit `624fa02d3e82db53895cec083248ba1b03554eb4`.

The client training and server parameter-update scaffolding in `fedckd/client.py`
and `fedckd/server.py` derive from the TFLlib-based FedCKD implementation.
This edition extracts the algorithm components and removes experiment drivers,
framework base classes, datasets, attack simulation, logging, and evaluation.

## PFLlib

Source: https://github.com/TsingZ0/PFLlib
License: Apache License 2.0, reproduced in `LICENSE`.
Source and license previously inspected at commit `0169ba7e412c9856a08bb3faefab1e35f538a3c1`.

`fedckd/model.py` follows the FedAvgCNN architecture in
`system/flcore/trainmodel/models.py`, with optional hidden-feature output.
This backbone is a component of FedCKD; no separate baseline method is included.

No dataset files or installed dependencies are bundled.
