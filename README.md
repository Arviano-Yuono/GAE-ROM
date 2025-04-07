# GAE-ROM: Graph Autoencoder for Reduced Order Modeling of Navier-Stokes Equations

A PyTorch implementation of a Graph Autoencoder (GAE) for Reduced Order Modeling (ROM) of Navier-Stokes equations on unstructured meshes. This project provides an efficient framework for learning reduced-order representations of fluid flow simulations using graph neural networks.

## Features

- Graph-based autoencoder architecture for handling unstructured mesh data
- Support for various graph convolution types (GMMConv, ChebConv, GCNConv, GATConv)
- Efficient data loading and preprocessing for Navier-Stokes solutions
- Configurable training pipeline with validation and checkpointing
- Visualization utilities for comparing full-order and reduced-order solutions
- Support for both 2D and 3D mesh data
- Automatic mesh-to-graph conversion utilities
- Comprehensive metrics for evaluating ROM performance

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/gae-rom.git
cd gae-rom

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package and its dependencies
pip install -e .
```

## Dependencies

The project requires the following main dependencies:
- PyTorch (>=2.0.0)
- PyTorch Geometric (>=2.3.0)
- NumPy (>=1.21.0)
- SciPy (>=1.7.0)
- Matplotlib (>=3.4.0)
- Pandas (>=1.3.0)
- scikit-learn (>=0.24.0)
- meshio (>=5.0.0)
- PyVista (>=0.39.0)
- VTK (>=9.0.0)
- h5py (>=3.0.0)

## Project Structure

```
GAE-ROM/
├── configs/                 # Configuration files
│   └── default.yaml        # Default training configuration
├── dataset/                # Directory for storing datasets
├── output/                 # Directory for model outputs and results
├── src/                    # Source code
│   ├── data/              # Data loading and preprocessing
│   │   ├── loader.py      # Data loading utilities
│   │   └── transform.py   # Data transformation functions
│   ├── model/             # Model architecture
│   │   ├── autoencoder.py # Autoencoder implementation
│   │   ├── convolution_layers.py # Graph convolution layers
│   │   ├── decoder.py     # Decoder network
│   │   ├── encoder.py     # Encoder network
│   │   └── gae.py         # Main GAE model
│   ├── training/          # Training utilities
│   │   └── train.py       # Training pipeline
│   └── utils/             # Utility functions
│       ├── commons.py     # Common utilities
│       ├── metrics.py     # Evaluation metrics
│       ├── normalization.py # Data normalization
│       └── scaler.py      # Data scaling utilities
├── mesh_to_graph.py       # Mesh to graph conversion script
├── requirements.txt       # Project dependencies
├── setup.py              # Package installation script
└── test_training.ipynb   # Training example notebook
```

## Usage

### Data Preparation

1. Prepare your Navier-Stokes data in the required format:
   - `mesh.su2`: Mesh vertices and elements
   - `flow.vtk`: Velocity field data

2. Convert mesh data to graph format:
```python
from gae_rom.utils import mesh_to_graph

# Convert mesh to graph
graph_data = mesh_to_graph(mesh_path, velocity_path, pressure_path)
```

### Training

1. Configure your model and training parameters in `configs/default.yaml`

2. Train the model:
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

### Evaluation

```python
# Load trained model
model = GAETrainer.load_model("path/to/checkpoint.pt")

# Evaluate on test data
metrics = trainer.evaluate(test_data)
print(f"Test metrics: {metrics}")
```

## Development

- Run tests: `pytest tests/`
- Format code: `black .`
- Sort imports: `isort .`
- Type checking: `mypy .`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the LICENSE file for details.