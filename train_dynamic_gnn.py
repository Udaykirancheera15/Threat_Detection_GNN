#!/usr/bin/env python3
"""
Dynamic GNN Model for Network Threat Detection
---------------------------------------------
This script implements and trains a Dynamic Graph Neural Network (GNN)
for detecting network threats based on GNS3 simulation data.

Usage:
    python train_dynamic_gnn.py --dataset [path] --output-dir [path]
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import time
import datetime
import random
from typing import Dict, List, Tuple, Union, Optional, Any

# Try to import PyTorch and related libraries
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool
    HAS_TORCH = True
except ImportError:
    print("Warning: PyTorch or PyTorch Geometric not installed.")
    print("Please install with: pip install torch torch_geometric")
    HAS_TORCH = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dynamic_gnn_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("dynamic_gnn_training")

if HAS_TORCH:
    class TemporalGraphDataset(Dataset):
        """PyTorch Dataset for temporal graph data."""
        
        def __init__(self, sequences: List[Dict]):
            """
            Initialize temporal graph dataset.
            
            Args:
                sequences: List of graph sequence dictionaries
            """
            self.sequences = sequences
            
        def __len__(self) -> int:
            return len(self.sequences)
        
        def __getitem__(self, idx: int) -> Tuple[Dict, int]:
            sequence = self.sequences[idx]
            return sequence["sequence"], sequence["label"]


    class DynamicGNN(nn.Module):
        """Dynamic Graph Neural Network for threat detection."""
        
        def __init__(self, input_dim: int, hidden_channels: int, num_classes: int = 2, 
                    num_layers: int = 3, dropout: float = 0.2):
            """
            Initialize dynamic GNN model.
            
            Args:
                input_dim: Input feature dimension
                hidden_channels: Number of hidden units
                num_classes: Number of output classes
                num_layers: Number of GNN layers
                dropout: Dropout probability
            """
            super(DynamicGNN, self).__init__()
            
            self.input_dim = input_dim
            self.hidden_channels = hidden_channels
            self.num_classes = num_classes
            self.num_layers = num_layers
            self.dropout = dropout
            
            # Initial feature transformation
            self.feature_transform = nn.Linear(input_dim, hidden_channels)
            
            # GNN layers
            self.convs = nn.ModuleList()
            self.batch_norms = nn.ModuleList()
            
            # First layer
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
            
            # Additional layers
            for _ in range(num_layers - 1):
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
                self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
            
            # Temporal attention layer
            self.temporal_attention = nn.Linear(hidden_channels, 1)
            
            # Output classification layer
            self.classifier = nn.Linear(hidden_channels, num_classes)
            
        def forward(self, x, edge_index, batch=None):
            """
            Main forward method required by PyTorch.
            
            Args:
                x: Either a single tensor or a list of tensors for temporal data
                edge_index: Either a single tensor or a list of tensors for temporal data
                batch: Optional batch vector
                
            Returns:
                Output predictions
            """
            # If input is a list, handle as temporal data
            if isinstance(x, list):
                return self.forward_temporal(x, edge_index, batch)
            else:
                # Process node embeddings
                node_embeddings = self.forward_single_graph(x, edge_index, batch)
                
                # Graph-level readout
                if batch is not None:
                    pooled = global_mean_pool(node_embeddings, batch)
                else:
                    pooled = torch.mean(node_embeddings, dim=0, keepdim=True)
                
                # Classification
                output = self.classifier(pooled)
                return output
                
        def forward_single_graph(self, x: torch.Tensor, edge_index: torch.Tensor, 
                              batch: Optional[torch.Tensor] = None) -> torch.Tensor:
            """
            Forward pass for a single graph.
            
            Args:
                x: Node feature matrix
                edge_index: Graph connectivity in COO format
                batch: Batch vector for multiple graphs (None for single graph)
                
            Returns:
                Node embeddings
            """
            # Initial feature transformation
            x = self.feature_transform(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # GNN layers
            for i in range(self.num_layers):
                x = self.convs[i](x, edge_index)
                x = self.batch_norms[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            
            return x
        
        def forward_temporal(self, 
                   x_sequence: List[torch.Tensor], 
                   edge_index_sequence: List[torch.Tensor], 
                   batch_sequence: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
            """
            Forward pass for a temporal sequence of graphs.
            
            Args:
                x_sequence: List of node feature matrices
                edge_index_sequence: List of graph connectivity in COO format
                batch_sequence: List of batch vectors (None for single graphs)
                
            Returns:
                Output predictions
            """
            sequence_length = len(x_sequence)
            
            # Process each graph in the sequence
            embeddings = []
            for t in range(sequence_length):
                x = x_sequence[t]
                edge_index = edge_index_sequence[t]
                batch = batch_sequence[t] if batch_sequence is not None else None
                
                # Process single graph
                node_embeddings = self.forward_single_graph(x, edge_index, batch)
                
                # Graph-level readout
                if batch is not None:
                    pooled_x = global_mean_pool(node_embeddings, batch)
                else:
                    pooled_x = torch.mean(node_embeddings, dim=0, keepdim=True)
                
                embeddings.append(pooled_x)
            
            # Stack embeddings from all time steps
            embeddings = torch.stack(embeddings, dim=1)  # Shape: [batch_size=1, sequence_length, hidden_channels]
            
            # Apply temporal attention
            attention_scores = self.temporal_attention(embeddings)  # Shape: [batch_size=1, sequence_length, 1]
            attention_weights = F.softmax(attention_scores, dim=1)  # Shape: [batch_size=1, sequence_length, 1]
            
            # Weighted sum of embeddings
            context = torch.sum(attention_weights * embeddings, dim=1)  # Shape: [batch_size=1, hidden_channels]
            
            # Classification
            output = self.classifier(context)
            
            return output

    class EvolveGCN(nn.Module):
        """
        Implementation of EvolveGCN model for temporal graph learning.
        
        Based on "EvolveGCN: Evolving Graph Convolutional Networks for Dynamic Graphs"
        """
        
        def __init__(self, input_dim: int, hidden_channels: int, num_classes: int = 2, 
                    num_layers: int = 2, dropout: float = 0.2):
            """
            Initialize EvolveGCN model.
            
            Args:
                input_dim: Input feature dimension
                hidden_channels: Number of hidden units
                num_classes: Number of output classes
                num_layers: Number of GNN layers
                dropout: Dropout probability
            """
            super(EvolveGCN, self).__init__()
            
            self.input_dim = input_dim
            self.hidden_channels = hidden_channels
            self.num_classes = num_classes
            self.num_layers = num_layers
            self.dropout = dropout
            
            # Initial feature transformation
            self.feature_transform = nn.Linear(input_dim, hidden_channels)
            
            # GRU cells to evolve the weights of GCN layers
            self.gru_cells = nn.ModuleList()
            
            # GCN weights for each layer
            self.weights = nn.ParameterList()
            
            # Initialize weights and GRU cells
            for i in range(num_layers):
                in_channels = hidden_channels
                out_channels = hidden_channels
                
                # Weight matrix parameter (equivalent to GCN weight matrix)
                weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
                nn.init.xavier_uniform_(weight)
                self.weights.append(weight)
                
                # GRU cell for evolving the weights
                self.gru_cells.append(nn.GRUCell(out_channels, in_channels))
            
            # Batch normalization layers
            self.batch_norms = nn.ModuleList()
            for _ in range(num_layers):
                self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
            
            # Output classification layer
            self.classifier = nn.Linear(hidden_channels, num_classes)
            
            
        
        def forward(self, x_sequence: List[torch.Tensor], 
                   edge_index_sequence: List[torch.Tensor], 
                   batch_sequence: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
            """
            Forward pass for a temporal sequence of graphs.
            
            Args:
                x_sequence: List of node feature tensors
                edge_index_sequence: List of edge index tensors
                batch_sequence: List of batch index tensors
                
            Returns:
                Output predictions
            """
            sequence_length = len(x_sequence)
            
            # Initial feature transformation for each graph in the sequence
            transformed_x_sequence = []
            for t in range(sequence_length):
                transformed_x_sequence.append(self.feature_transform(x_sequence[t]))
            
            # Process each layer
            h_sequence = transformed_x_sequence
            
            for l in range(self.num_layers):
                # Get initial weight for this layer
                weight = self.weights[l]
                
                # Create hidden state for GRU (for each layer)
                h = weight.view(1, -1)  # Shape: [1, in_channels * out_channels]
                
                layer_output_sequence = []
                
                # Process the sequence for this layer
                for t in range(sequence_length):
                    # Get current graph
                    x = h_sequence[t]
                    edge_index = edge_index_sequence[t]
                    
                    # Use current weight matrix for GCN operation
                    weight_reshaped = h.view(weight.shape)
                    
                    # Simple GCN message passing
                    from torch_geometric.utils import degree
                    row, col = edge_index
                    deg = degree(row, x.size(0), dtype=x.dtype)
                    deg_inv_sqrt = deg.pow(-0.5)
                    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
                    norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
                    
                    # GCN convolution
                    support = torch.mm(x, weight_reshaped)
                    aggregated = torch.zeros_like(support)
                    aggregated.index_add_(0, row, support[col] * norm.view(-1, 1))
                    
                    # Apply batch normalization
                    output = self.batch_norms[l](aggregated)
                    
                    # Nonlinearity
                    output = F.relu(output)
                    
                    # Apply dropout
                    output = F.dropout(output, p=self.dropout, training=self.training)
                    
                    # Store output for this timestep
                    layer_output_sequence.append(output)
                    
                    # Update the weight using GRU
                    if t < sequence_length - 1:
                        # We take the mean of the node embeddings as the "input" to the GRU
                        input_to_gru = torch.mean(output, dim=0, keepdim=True)  # Shape: [1, hidden_channels]
                        h = self.gru_cells[l](input_to_gru, h)  # Shape: [1, in_channels * out_channels]
                
                # Update h_sequence for the next layer
                h_sequence = layer_output_sequence
            
            # Final graph readout from the last graph in the sequence
            final_x = h_sequence[-1]
            final_batch = batch_sequence[-1] if batch_sequence is not None else None
            
            if final_batch is not None:
                pooled = global_mean_pool(final_x, final_batch)
            else:
                pooled = torch.mean(final_x, dim=0, keepdim=True)
            
            # Classification
            output = self.classifier(pooled)
            
            return output


    class ModelTrainer:
        """Trainer for dynamic graph neural network models."""
        
        def __init__(self, model: nn.Module, config: Dict = None):
            """
            Initialize the model trainer.
            
            Args:
                model: PyTorch model to train
                config: Configuration dictionary
            """
            self.model = model
            
            # Default configuration
            self.config = {
                "use_gpu": torch.cuda.is_available(),
                "learning_rate": 0.001,
                "weight_decay": 5e-4,
                "batch_size": 32,
                "epochs": 50,
                "patience": 10,
                "checkpoint_dir": "./models/checkpoints"
            }
            
            # Update with provided config
            if config:
                self.config.update(config)
            
            # Device setup
            self.device = torch.device('cuda' if self.config['use_gpu'] and torch.cuda.is_available() else 'cpu')
            logger.info(f"Using device: {self.device}")
            
            # Move model to device
            self.model = self.model.to(self.device)
            
            # Initialize optimizer
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay']
            )
            
            # Learning rate scheduler
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
            )
            
            # Track best model
            self.best_val_loss = float('inf')
            self.best_model_state = None
            self.no_improve_count = 0
            
            # Create checkpoint directory
            os.makedirs(self.config['checkpoint_dir'], exist_ok=True)
        
        def prepare_batch(self, batch: Tuple[List[Dict], List[int]]) -> Tuple[List[torch.Tensor], List[torch.Tensor], Optional[List[torch.Tensor]], torch.Tensor]:
            """
            Prepare a batch of temporal graph sequences for input to the model.
            
            Args:
                batch: Tuple of (sequences, labels)
                
            Returns:
                Tuple of (x_sequence, edge_index_sequence, batch_sequence, labels)
            """
            sequences, labels = batch
            
            # Convert labels to tensor
            labels_tensor = torch.tensor(labels, dtype=torch.long).to(self.device)
            
            # Prepare sequences
            x_sequence = []
            edge_index_sequence = []
            batch_sequence = None  # Not used in this implementation
            
            # With batch size 1, we're processing only one sequence
            sequence = sequences[0]  # Get the only sequence in the batch
            
            for t in range(len(sequence)):  # For each timestep
                graph = sequence[t]  # Graph at timestep t
                
                # Convert node features to tensor
                x = torch.tensor(graph['node_features'], dtype=torch.float).to(self.device)
                x_sequence.append(x)
                
                # Convert edge index to tensor
                edge_index = torch.tensor(graph['edge_index'], dtype=torch.long).to(self.device)
                edge_index_sequence.append(edge_index)
            
            return x_sequence, edge_index_sequence, batch_sequence, labels_tensor
        
        def train_epoch(self, train_loader: DataLoader) -> Dict:
            """
            Train the model for one epoch.
            
            Args:
                train_loader: DataLoader with training data
                
            Returns:
                Dictionary with training metrics
            """
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_idx, batch in enumerate(train_loader):
                # Prepare batch data
                x_sequence, edge_index_sequence, batch_sequence, labels = self.prepare_batch(batch)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(x_sequence, edge_index_sequence, batch_sequence)
                
                # Compute loss
                loss = F.cross_entropy(outputs, labels)
                
                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()
                
                # Update statistics
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Print progress
                if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(train_loader):
                    logger.info(f'Train Epoch: {batch_idx+1}/{len(train_loader)} '
                               f'Loss: {total_loss/(batch_idx+1):.4f} '
                               f'Acc: {100.*correct/total:.2f}%')
            
            # Compute final metrics
            avg_loss = total_loss / len(train_loader)
            accuracy = 100. * correct / total
            
            return {
                'loss': avg_loss,
                'accuracy': accuracy
            }
        
        def validate(self, val_loader: DataLoader) -> Dict:
            """
            Validate the model on the validation set.
            
            Args:
                val_loader: DataLoader with validation data
                
            Returns:
                Dictionary with validation metrics
            """
            self.model.eval()
            val_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    # Prepare batch data
                    x_sequence, edge_index_sequence, batch_sequence, labels = self.prepare_batch(batch)
                    
                    # Forward pass
                    outputs = self.model(x_sequence, edge_index_sequence, batch_sequence)
                    
                    # Compute loss
                    loss = F.cross_entropy(outputs, labels)
                    
                    # Update statistics
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
            
            # Compute final metrics
            avg_loss = val_loss / len(val_loader)
            accuracy = 100. * correct / total
            
            logger.info(f'Validation Loss: {avg_loss:.4f} Acc: {accuracy:.2f}%')
            
            return {
                'loss': avg_loss,
                'accuracy': accuracy
            }
        
        def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int = None) -> Dict:
            """
            Train the model for multiple epochs.
            
            Args:
                train_loader: DataLoader with training data
                val_loader: DataLoader with validation data
                epochs: Number of epochs to train for (use config value if None)
                
            Returns:
                Dictionary with training history
            """
            if epochs is None:
                epochs = self.config['epochs']
            
            patience = self.config['patience']
            
            logger.info(f"Starting training for {epochs} epochs...")
            
            history = {
                'train_loss': [],
                'train_acc': [],
                'val_loss': [],
                'val_acc': [],
                'best_epoch': 0
            }
            
            for epoch in range(epochs):
                logger.info(f"Epoch {epoch+1}/{epochs}")
                
                # Train for one epoch
                train_metrics = self.train_epoch(train_loader)
                
                # Validate
                val_metrics = self.validate(val_loader)
                
                # Update learning rate
                self.scheduler.step(val_metrics['loss'])
                
                # Save metrics
                history['train_loss'].append(train_metrics['loss'])
                history['train_acc'].append(train_metrics['accuracy'])
                history['val_loss'].append(val_metrics['loss'])
                history['val_acc'].append(val_metrics['accuracy'])
                
                # Check for improvement
                if val_metrics['loss'] < self.best_val_loss:
                    logger.info(f"Validation loss improved from {self.best_val_loss:.4f} to {val_metrics['loss']:.4f}")
                    self.best_val_loss = val_metrics['loss']
                    self.best_model_state = self.model.state_dict()
                    self.no_improve_count = 0
                    history['best_epoch'] = epoch
                    
                    # Save best model checkpoint
                    self.save_checkpoint(is_best=True)
                else:
                    self.no_improve_count += 1
                    logger.info(f"No improvement in validation loss for {self.no_improve_count} epochs")
                    
                    # Early stopping
                    if self.no_improve_count >= patience:
                        logger.info(f"Early stopping triggered after {epoch+1} epochs")
                        break
                
                # Save regular checkpoint every 5 epochs
                if (epoch + 1) % 5 == 0:
                    self.save_checkpoint()
            
            # Load best model
            if self.best_model_state is not None:
                self.model.load_state_dict(self.best_model_state)
                logger.info("Loaded best model based on validation loss")
            
            return history
        
        def save_checkpoint(self, is_best: bool = False) -> str:
            """
            Save model checkpoint.
            
            Args:
                is_best: Whether this is the best model so far
                
            Returns:
                Path to saved checkpoint
            """
            # Create timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create checkpoint filename
            prefix = "best_" if is_best else ""
            model_name = self.model.__class__.__name__
            checkpoint_file = os.path.join(self.config['checkpoint_dir'], f"{prefix}{model_name}_{timestamp}.pt")
            
            # Create checkpoint dictionary
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_val_loss': self.best_val_loss,
                'config': self.config,
                'model_name': model_name,
                'timestamp': timestamp
            }
            
            # Save checkpoint
            torch.save(checkpoint, checkpoint_file)
            logger.info(f"Saved {'best ' if is_best else ''}checkpoint to {checkpoint_file}")
            
            return checkpoint_file
        
        def evaluate(self, test_loader: DataLoader) -> Dict:
            """
            Evaluate the model on the test set.
            
            Args:
                test_loader: DataLoader with test data
                
            Returns:
                Dictionary with evaluation metrics
            """
            self.model.eval()
            test_loss = 0
            correct = 0
            total = 0
            
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(test_loader):
                    # Prepare batch data
                    x_sequence, edge_index_sequence, batch_sequence, labels = self.prepare_batch(batch)
                    
                    # Forward pass
                    outputs = self.model(x_sequence, edge_index_sequence, batch_sequence)
                    
                    # Compute loss
                    loss = F.cross_entropy(outputs, labels)
                    
                    # Update statistics
                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
                    
                    # Store predictions and labels for metrics calculation
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            # Compute final metrics
            avg_loss = test_loss / len(test_loader)
            accuracy = 100. * correct / total
            
            # Calculate additional metrics
            from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_auc_score
            
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, all_preds, average='binary', pos_label=1
            )
            
            cm = confusion_matrix(all_labels, all_preds)
            
            # Try to calculate AUC if we have probability outputs
            try:
                roc_auc = roc_auc_score(all_labels, all_preds)
            except:
                roc_auc = None
            
            logger.info(f'Test Loss: {avg_loss:.4f} Acc: {accuracy:.2f}%')
            logger.info(f'Precision: {precision:.4f} Recall: {recall:.4f} F1: {f1:.4f}')
            logger.info(f'Confusion Matrix: \n{cm}')
            if roc_auc:
                logger.info(f'ROC AUC: {roc_auc:.4f}')
            
            return {
                'loss': avg_loss,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'confusion_matrix': cm.tolist(),
                'roc_auc': roc_auc
            }


def train_dynamic_gnn(dataset_path: str, output_dir: str, config: Dict = None):
    """
    Train a Dynamic GNN model for threat detection.
    
    Args:
        dataset_path: Path to processed dataset
        output_dir: Directory to save model and results
        config: Configuration dictionary
        
    Returns:
        Dictionary with training results
    """
    if not HAS_TORCH:
        logger.error("PyTorch is required for training")
        return {"error": "PyTorch not available"}
    
    # Default configuration
    if config is None:
        config = {
            "model_type": "DynamicGNN",  # or "EvolveGCN"
            "hidden_channels": 64,
            "num_layers": 3,
            "dropout": 0.2,
            "learning_rate": 0.001,
            "weight_decay": 5e-4,
            "batch_size": 32,
            "epochs": 50,
            "patience": 10,
            "use_gpu": torch.cuda.is_available()
        }
    
    # Load dataset
    logger.info(f"Loading dataset from {dataset_path}")
    try:
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
    except Exception as e:
        logger.error(f"Failed to load dataset: {str(e)}")
        return {"error": f"Failed to load dataset: {str(e)}"}
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create data loaders
    train_dataset = TemporalGraphDataset(dataset['train'])
    val_dataset = TemporalGraphDataset(dataset['validation'])
    test_dataset = TemporalGraphDataset(dataset['test'])
        
        # Define a custom collate function to handle varying graph sizes
    def custom_collate(batch):
        # Batch is a list of (sequence, label) tuples
        sequences = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        return sequences, labels

    # Create data loaders with the custom collate function and smaller batch size
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=custom_collate)
    test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=custom_collate)
    
    logger.info(f"Created data loaders: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    # Determine input dimension (from first graph in training set)
    first_sequence = dataset['train'][0]['sequence']
    first_graph = first_sequence[0]
    input_dim = first_graph['node_features'].shape[1]
    
    # Create model
    logger.info(f"Creating {config['model_type']} model")
    
    if config['model_type'] == "DynamicGNN":
        model = DynamicGNN(
            input_dim=input_dim,
            hidden_channels=config['hidden_channels'],
            num_layers=config['num_layers'],
            dropout=config['dropout']
        )
    elif config['model_type'] == "EvolveGCN":
        model = EvolveGCN(
            input_dim=input_dim,
            hidden_channels=config['hidden_channels'],
            num_layers=config['num_layers'],
            dropout=config['dropout']
        )
    else:
        logger.error(f"Unsupported model type: {config['model_type']}")
        return {"error": f"Unsupported model type: {config['model_type']}"}
    
    # Create trainer
    trainer_config = {
        "learning_rate": config['learning_rate'],
        "weight_decay": config['weight_decay'],
        "batch_size": config['batch_size'],
        "epochs": config['epochs'],
        "patience": config['patience'],
        "use_gpu": config['use_gpu'],
        "checkpoint_dir": os.path.join(output_dir, "checkpoints")
    }
    
    trainer = ModelTrainer(model, trainer_config)
    
    # Train model
    logger.info("Starting model training")
    start_time = time.time()
    history = trainer.train(train_loader, val_loader)
    training_time = time.time() - start_time
    
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    # Evaluate on test set
    logger.info("Evaluating model on test set")
    test_metrics = trainer.evaluate(test_loader)
    
    # Save results
    results = {
        "config": config,
        "history": history,
        "test_metrics": test_metrics,
        "training_time": training_time,
        "model_type": config['model_type'],
        "input_dim": input_dim,
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    results_file = os.path.join(output_dir, f"results_{config['model_type']}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    with open(results_file, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        for key, value in results.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        results[key][k] = v.tolist()
        
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {results_file}")
    
    return results


def main():
    """Main function to train a Dynamic GNN model."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train Dynamic GNN for Network Threat Detection")
    parser.add_argument("--dataset", type=str, required=True, help="Path to processed dataset")
    parser.add_argument("--output-dir", type=str, default="./models", help="Directory to save model and results")
    parser.add_argument("--model-type", type=str, default="DynamicGNN", choices=["DynamicGNN", "EvolveGCN"], help="Model type")
    parser.add_argument("--hidden-channels", type=int, default=64, help="Number of hidden channels")
    parser.add_argument("--num-layers", type=int, default=3, help="Number of GNN layers")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout probability")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="Weight decay")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU usage")
    
    args = parser.parse_args()
    
    # Check if dataset exists
    if not os.path.exists(args.dataset):
        logger.error(f"Dataset not found: {args.dataset}")
        return
    
    # Create configuration from arguments
    config = {
        "model_type": args.model_type,
        "hidden_channels": args.hidden_channels,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "use_gpu": not args.no_gpu and torch.cuda.is_available()
    }
    
    # Train model
    results = train_dynamic_gnn(args.dataset, args.output_dir, config)
    
    if "error" in results:
        logger.error(f"Training failed: {results['error']}")
    else:
        print("\nModel training completed successfully!")
        print(f"Best validation accuracy: {max(results['history']['val_acc']):.2f}% (epoch {results['history']['best_epoch']+1})")
        print(f"Test accuracy: {results['test_metrics']['accuracy']:.2f}%")
        print(f"Test F1 score: {results['test_metrics']['f1']:.4f}")
        print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    if not HAS_TORCH:
        print("Error: PyTorch and PyTorch Geometric are required for training.")
        print("Please install with: pip install torch torch_geometric")
        sys.exit(1)
    
    main()
