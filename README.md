# Dynamic GNN Network Threat Detection

This project implements a Dynamic Graph Neural Network (GNN) model for network threat detection using GNS3 simulations. It transforms network traffic data into graph representations and applies dynamic graph neural networks to detect potential security threats in real-time.

## Features

- **GNS3 Integration**: Connect to GNS3 environments to capture and analyze real network traffic
- **Synthetic Data Generation**: Generate synthetic network traffic data when no GNS3 environment is available
- **Dynamic Graph Neural Networks**: Implement state-of-the-art graph neural network models for temporal graph analysis
- **Real-time Threat Detection**: Monitor network traffic in real-time and alert on potential threats
- **Complete Workflow**: End-to-end pipeline from data generation to model training and deployment

## Project Structure

```
dynamic-gnn-threat-detection/
├── dynamic_gnn_threat_detection.py  # Main threat detection system
├── monitor_gns3.py                  # GNS3 monitoring integration
├── setup_gns3_simulation.py         # GNS3 simulation setup
├── preprocess_data.py               # Data preprocessing for Dynamic GNN
├── train_dynamic_gnn.py             # Dynamic GNN model implementation and training
├── gnn_threat_detector.py           # Threat detector integration
├── run_project.py                   # Complete project workflow
├── data/                            # Data directory
│   ├── gns3_simulation/             # GNS3 simulation data
│   └── processed/                   # Processed data for training
├── models/                          # Trained models
│   └── checkpoints/                 # Model checkpoints
└── results/                         # Evaluation results
```

## Requirements

- Python 3.7+
- PyTorch 1.8+
- PyTorch Geometric
- Pandas
- NumPy
- GNS3 (optional, for real network simulation)

Install the required packages:

```bash
pip install torch torch_geometric pandas numpy
```

For GNS3 integration:

```bash
pip install gns3fy requests
```

## Quick Start

### Complete Workflow

Run the complete workflow from data generation to model evaluation:

```bash
python run_project.py --simulate
```

This will:
1. Generate synthetic network traffic data (with normal and attack patterns)
2. Preprocess the data for Dynamic GNN training
3. Train a Dynamic GNN model
4. Evaluate the model on test data

### Monitoring Mode

Run the trained model in monitoring mode to detect threats in real-time:

```bash
python gnn_threat_detector.py --model models/checkpoints/best_DynamicGNN_XXXXXXXX.pt --monitor
```

### Processing Specific Captures

Process specific capture files or directories:

```bash
python gnn_threat_detector.py --model models/checkpoints/best_DynamicGNN_XXXXXXXX.pt --process data/captures/
```

## Detailed Usage

### GNS3 Simulation Setup

```bash
python setup_gns3_simulation.py [--simulate] [--duration SECONDS] [--attack-duration SECONDS]
```

Options:
- `--simulate`: Use simulated data instead of GNS3
- `--duration`: Duration of normal traffic simulation (default: 600 seconds)
- `--attack-duration`: Duration of attack simulation (default: 300 seconds)

### Data Preprocessing

```bash
python preprocess_data.py --data-dir data/gns3_simulation --window-size 5
```

Options:
- `--data-dir`: Directory containing network capture data
- `--window-size`: Size of temporal window for sequence creation (default: 5)

### Model Training

```bash
python train_dynamic_gnn.py --dataset data/processed/dataset.pkl --model-type DynamicGNN
```

Options:
- `--dataset`: Path to preprocessed dataset
- `--model-type`: Model type (DynamicGNN or EvolveGCN)
- `--hidden-channels`: Number of hidden channels (default: 64)
- `--num-layers`: Number of GNN layers (default: 3)
- `--epochs`: Maximum number of training epochs (default: 50)

### GNN Threat Detector

```bash
python gnn_threat_detector.py --model models/checkpoints/best_model.pt [--monitor | --process PATH]
```

Options:
- `--model`: Path to trained model checkpoint
- `--monitor`: Run in continuous monitoring mode
- `--process`: Process a specific capture file or directory
- `--threshold`: Threat detection threshold (default: 0.7)

## Model Architecture

This project implements two types of dynamic graph neural networks:

1. **DynamicGNN**: A custom GNN architecture with temporal attention mechanism
2. **EvolveGCN**: Implementation of the EvolveGCN model from "EvolveGCN: Evolving Graph Convolutional Networks for Dynamic Graphs"

The models process sequences of network traffic graphs to detect patterns indicative of security threats.

## GNS3 Integration

The project can integrate with GNS3 for:
- Collecting real network traffic data
- Simulating network attacks
- Monitoring live network environments

When no GNS3 environment is available, the system can fall back to synthetic data generation.

## Results and Evaluation

The system evaluates threat detection using standard metrics:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC

Results are saved in the `results/` directory, including:
- Model performance metrics
- Threat detection statistics
- Visualization data

## Advanced Configuration

Fine-tune the system by modifying configuration parameters in the scripts:

- **Graph Encoding**: Adjust node and edge feature extraction in `GraphEncoder` class
- **Model Architecture**: Modify model hyperparameters in training scripts
- **GNS3 Integration**: Configure GNS3 connection parameters in `monitor_gns3.py`
- **Threat Detection**: Adjust detection thresholds in `gnn_threat_detector.py`

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- GNS3 team for the network simulation platform
- PyTorch and PyTorch Geometric teams for the deep learning frameworks
- The research community for advancements in dynamic graph neural networks
