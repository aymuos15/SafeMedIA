# DP-FedMed

Privacy-preserving federated learning framework for medical image segmentation with optional SSL pretraining.

## Installation

```bash
pip install -e .
```

## Quick Start

### Federated Learning

1. Set data directory in `configs/default.toml`:
   ```toml
   [data]
   data_dir = "/path/to/Dataset001_Cellpose"
   ```

2. Run:
   ```bash
   flwr run .
   ```

### SSL Pretraining (Optional)

For transfer learning, pretrain on unlabeled data first:

1. Update `configs/pretraining.toml` with your unlabeled data directory
2. Run:
   ```bash
   python scripts/pretrain.py --config configs/pretraining.toml
   ```

3. Use pretrained encoder in `configs/default.toml`:
   ```toml
   [ssl]
   pretrained_checkpoint_path = "checkpoints/pretrained_encoder.pt"
   freeze_encoder = false
   ```

4. Run federated learning as above

## Dataset Structure

```
Dataset001_Cellpose/
├── imagesTr/
│   ├── 000_0000.png
│   └── ...
├── labelsTr/
│   ├── 000.png
│   └── ...
├── imagesTs/
└── labelsTs/
```

## Configuration

Default settings in `configs/default.toml`:
- Clients: 2
- Rounds: 5
- Local epochs: 5
- Batch size: 8
- Privacy style: user-level DP

SSL methods: `simclr` (default), `moco`, `simsiam`

## Results

```
results/
└── default/
    ├── server/
    │   ├── train.log
    │   ├── metrics.json
    │   └── history.json
    ├── client_0/
    └── client_1/
```

## Testing

```bash
pytest tests/test_ssl_pretraining.py -v
pytest tests/test_pretrained_federated.py -v
```
