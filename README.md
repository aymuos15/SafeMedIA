# DP-FedMed

Privacy-preserving federated learning framework for medical image segmentation using Flower, Opacus, and MONAI.

## Dataset Structure

```
Dataset001_Cellpose/
├── imagesTr/
│   ├── 000_0000.png
│   ├── 001_0000.png
│   └── ...
├── labelsTr/
│   ├── 000.png
│   ├── 001.png
│   └── ...
├── imagesTs/
│   └── ...
└── labelsTs/
    └── ...
```

## Default Configuration

| Setting | Value |
|---------|-------|
| Clients | 2 |
| Rounds | 5 |
| Local epochs | 5 |
| Batch size | 8 |
| Target ε | 8.0 |
| Noise multiplier | 1.0 |
| GPU per client | 0.3 |

## Running

1. Install the package:
   ```bash
   pip install -e .
   ```

2. Set your data directory in `configs/default.toml`:
   ```toml
   [data]
   data_dir = "/path/to/Dataset001_Cellpose"
   ```

3. Run:
   ```bash
   flwr run .
   ```

## Results

```
results/
└── default/
    ├── server/
    │   ├── train.log
    │   ├── metrics.json
    │   └── history.json
    ├── client_0/
    │   ├── train.log
    │   ├── metrics.json
    │   └── history.json
    └── client_1/
        ├── train.log
        ├── metrics.json
        └── history.json
```
