# GAE-ROM: Graph Autoencoder for Reduced Order Modeling of Navier-Stokes Equations

A PyTorch implementation of a Graph Autoencoder (GAE) for Reduced Order Modeling (ROM) of Navier-Stokes equations on unstructured meshes.

## Features

- Graph-based autoencoder architecture for handling unstructured mesh data
- Support for various graph convolution types (GMMConv, ChebConv, GCNConv, GATConv)
- Efficient data loading and preprocessing for Navier-Stokes solutions
- Configurable training pipeline with validation and checkpointing
- Visualization utilities for comparing full-order and reduced-order solutions

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/gae-rom.git
cd gae-rom

# Install the package
pip install -e .

# Install development dependencies
pip install -e ".[dev]"
```

## Project Structure

```
GAE-ROM/
├── .git/
├── configs/
│   └── default.yaml
├── dataset/
├── output/
├── src/
│   ├── data/
│   │   ├── __pycache__/
│   │   ├── loader.py
│   │   └── transform.py
│   ├── model/
│   │   ├── autoencoder.py
│   │   ├── convolution_layers.py
│   │   ├── decoder.py
│   │   ├── encoder.py
│   │   └── gae.py
│   ├── training/
│   │   └── train.py
│   └── utils/
│       ├── __pycache__/
│       ├── __init__.py
│       ├── commons.py
│       ├── metrics.py
│       ├── normalization.py
│       └── scaler.py
├── mesh_to_graph.py
├── README.md
├── requirements.txt
├── setup.py
└── test_training.ipynb
```

## Usage

1. Prepare your Navier-Stokes data in the required format:
   - `mesh.npy`: Mesh vertices and elements
   - `velocity.npy`: Velocity field data
   - `pressure.npy`: Pressure field data

2. Configure your model and training parameters in `configs/default.yaml`

3. Train the model:
```python
from gae_rom.training import GAETrainer
from gae_rom.config import load_config

# Load configuration
config = load_config("configs/default.yaml")

# Initialize trainer
trainer = GAETrainer(config)

# Train the model
trainer.train(num_epochs=100)
```

## Development

- Run tests: `pytest tests/`
- Format code: `black .`
- Sort imports: `isort .`
- Type checking: `mypy .`

## Contributing

Feel free to open issues or submit pull requests if you find any bugs or have suggestions for improvements.