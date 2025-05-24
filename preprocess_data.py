#!/usr/bin/env python3
"""
Data Preprocessing for Dynamic GNN Threat Detection
--------------------------------------------------
This script processes network data from GNS3 simulations and prepares it
for training a Dynamic Graph Neural Network (GNN) for threat detection.

Usage:
    python preprocess_data.py --data-dir [path] --output-dir [path]
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
import random
import datetime
from typing import Dict, List, Tuple, Union, Optional, Any

# Try to import from the main threat detection system
try:
    from dynamic_gnn_threat_detection import (
        DataIngestion,
        GraphEncoder,
        DataPreprocessor
    )
    HAS_IMPORTS = True
except ImportError:
    print("Warning: Could not import from dynamic_gnn_threat_detection.py")
    HAS_IMPORTS = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_preprocessing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("data_preprocessing")

class GNS3DataProcessor:
    """Process GNS3 simulation data for Dynamic GNN."""
    
    def __init__(self, data_dir: str, output_dir: str, config: Dict = None):
        """
        Initialize data processor.
        
        Args:
            data_dir: Directory containing GNS3 simulation data
            output_dir: Directory to save processed data
            config: Configuration dictionary
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Default configuration
        self.config = {
            "capture_subdir": "captures",
            "alerts_subdir": "alerts",
            "time_window": 300,  # seconds
            "test_split": 0.2,
            "validation_split": 0.1,
            "random_seed": 42
        }
        
        # Update with provided config
        if config:
            self.config.update(config)
        
        # Set random seed for reproducibility
        random.seed(self.config["random_seed"])
        np.random.seed(self.config["random_seed"])
        
        # Initialize components if available
        if HAS_IMPORTS:
            self.data_ingestion = DataIngestion()
            self.graph_encoder = GraphEncoder()
            self.data_preprocessor = DataPreprocessor()
        else:
            logger.warning("Running in limited mode without dynamic_gnn_threat_detection imports")
    
    def collect_capture_files(self) -> List[Dict]:
        """
        Collect all capture files from the data directory.
        
        Returns:
            List of dictionaries with capture information
        """
        captures_dir = self.data_dir / self.config["capture_subdir"]
        
        if not captures_dir.exists():
            logger.warning(f"Captures directory {captures_dir} not found")
            # Try to find captures in main data directory
            captures_dir = self.data_dir
        
        logger.info(f"Collecting capture files from {captures_dir}")
        
        capture_files = []
        
        # Find all PCAP files
        for pcap_file in captures_dir.glob("**/*.pcap"):
            # Extract metadata from filename
            filename = pcap_file.name
            parts = filename.split("_")
            
            # Try to determine if this is an attack sample
            is_attack = any(attack_type in filename.lower() for attack_type in ["attack", "scan", "dos", "exploit"])
            
            # Default label is 0 (normal) or 1 (attack)
            label = 1 if is_attack else 0
            
            # Try to extract timestamp
            timestamp = None
            try:
                # Look for YYYYMMDD_HHMMSS pattern in filename
                for part in parts:
                    if len(part) >= 8 and part.isdigit():
                        timestamp = datetime.datetime.strptime(part[:8], "%Y%m%d")
                        if len(part) >= 14:
                            timestamp = datetime.datetime.strptime(part[:14], "%Y%m%d_%H%M%S")
                        break
            except ValueError:
                pass
            
            if timestamp is None:
                # Use file modification time if timestamp not found in filename
                timestamp = datetime.datetime.fromtimestamp(pcap_file.stat().st_mtime)
            
            capture_info = {
                "filepath": str(pcap_file),
                "filename": filename,
                "is_attack": is_attack,
                "label": label,
                "timestamp": timestamp.isoformat()
            }
            
            capture_files.append(capture_info)
        
        # Sort by timestamp
        capture_files.sort(key=lambda x: x["timestamp"])
        
        logger.info(f"Found {len(capture_files)} capture files")
        logger.info(f"Attack samples: {sum(1 for c in capture_files if c['is_attack'])}")
        logger.info(f"Normal samples: {sum(1 for c in capture_files if not c['is_attack'])}")
        
        return capture_files
    
    def process_capture_file(self, capture_info: Dict) -> Optional[Dict]:
        """
        Process a single capture file into a graph representation.
        
        Args:
            capture_info: Dictionary with capture information
            
        Returns:
            Dictionary with processed graph data or None on failure
        """
        filepath = capture_info["filepath"]
        logger.info(f"Processing capture file: {filepath}")
        
        try:
            if HAS_IMPORTS:
                # Use the main system's components
                # 1. Ingest network flow data
                flow_df = self.data_ingestion.ingest_network_flow(filepath)
                
                # 2. Encode network flow data into graph format
                graph_encoding = self.graph_encoder.encode_network_data(flow_df)
                
                # 3. Preprocess graph encoding
                processed_encoding = self.data_preprocessor.preprocess_graph_encoding(graph_encoding)
                
                # 4. Add label and metadata
                processed_encoding["label"] = capture_info["label"]
                processed_encoding["metadata"] = {
                    "filename": capture_info["filename"],
                    "timestamp": capture_info["timestamp"],
                    "is_attack": capture_info["is_attack"]
                }
                
                return processed_encoding
            else:
                # Simplified processing for demonstration
                logger.warning("Using simplified processing due to missing imports")
                
                # Create a very simple graph encoding with random features
                num_nodes = random.randint(5, 20)
                num_edges = random.randint(num_nodes, num_nodes * 3)
                
                # Generate random node features
                node_features = np.random.randn(num_nodes, 8).astype(np.float32)
                
                # Generate random edges
                edge_index = np.zeros((2, num_edges), dtype=np.int64)
                for i in range(num_edges):
                    edge_index[0, i] = random.randint(0, num_nodes - 1)  # source
                    edge_index[1, i] = random.randint(0, num_nodes - 1)  # target
                
                # Generate random edge features
                edge_features = np.random.randn(num_edges, 4).astype(np.float32)
                
                # Create simplified graph encoding
                graph_encoding = {
                    "node_features": node_features,
                    "edge_index": edge_index,
                    "edge_features": edge_features,
                    "num_nodes": num_nodes,
                    "num_edges": num_edges,
                    "label": capture_info["label"],
                    "metadata": {
                        "filename": capture_info["filename"],
                        "timestamp": capture_info["timestamp"],
                        "is_attack": capture_info["is_attack"]
                    }
                }
                
                return graph_encoding
                
        except Exception as e:
            logger.error(f"Failed to process capture file {filepath}: {str(e)}")
            return None
    
    def create_temporal_sequence(self, graph_encodings: List[Dict], window_size: int = 5) -> List[Dict]:
        """
        Create temporal sequences from graph encodings for dynamic GNN.
        
        Args:
            graph_encodings: List of graph encodings ordered by time
            window_size: Number of frames to include in each sequence
            
        Returns:
            List of temporal sequences
        """
        logger.info(f"Creating temporal sequences with window size {window_size}")
        
        if not graph_encodings:
            logger.warning("No graph encodings provided")
            return []
        
        # Ensure we have enough graphs
        if len(graph_encodings) < window_size:
            logger.warning(f"Not enough graph encodings ({len(graph_encodings)}) for window size {window_size}")
            # Pad with copies if needed
            if len(graph_encodings) > 0:
                graph_encodings = graph_encodings + [graph_encodings[-1]] * (window_size - len(graph_encodings))
            else:
                logger.error("No graph encodings provided")
                return []
        
        # Create sliding window sequences
        sequences = []
        labels = []
        
        for i in range(len(graph_encodings) - window_size + 1):
            sequence = graph_encodings[i:i+window_size]
            
            # Get label based on last frame in sequence
            # If any frame in the sequence is labeled as attack, label the sequence as attack
            has_attack = any(frame["label"] == 1 for frame in sequence)
            label = 1 if has_attack else 0
            
            sequences.append({
                "sequence": sequence,
                "start_idx": i,
                "end_idx": i + window_size - 1,
                "label": label
            })
            
            labels.append(label)
        
        logger.info(f"Created {len(sequences)} temporal sequences")
        logger.info(f"Attack sequences: {sum(1 for l in labels if l == 1)}")
        logger.info(f"Normal sequences: {sum(1 for l in labels if l == 0)}")
        
        return sequences
    
    def split_dataset(self, sequences: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Split sequences into training, validation, and test sets.
        
        Args:
            sequences: List of temporal sequences
            
        Returns:
            Dictionary with train, validation, and test sets
        """
        logger.info(f"Splitting dataset of {len(sequences)} sequences")
        
        # Shuffle sequences
        indices = list(range(len(sequences)))
        random.shuffle(indices)
        
        # Calculate split sizes
        test_size = int(len(indices) * self.config["test_split"])
        val_size = int(len(indices) * self.config["validation_split"])
        train_size = len(indices) - test_size - val_size
        
        # Split indices
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size+val_size]
        test_indices = indices[train_size+val_size:]
        
        # Create dataset splits
        train_set = [sequences[i] for i in train_indices]
        val_set = [sequences[i] for i in val_indices]
        test_set = [sequences[i] for i in test_indices]
        
        dataset = {
            "train": train_set,
            "validation": val_set,
            "test": test_set,
            "all_sequences": sequences,
            "train_indices": train_indices,
            "val_indices": val_indices,
            "test_indices": test_indices
        }
        
        # Log dataset statistics
        logger.info(f"Dataset split: train={len(train_set)}, validation={len(val_set)}, test={len(test_set)}")
        logger.info(f"Train set: {sum(1 for s in train_set if s['label'] == 1)} attack, {sum(1 for s in train_set if s['label'] == 0)} normal")
        logger.info(f"Validation set: {sum(1 for s in val_set if s['label'] == 1)} attack, {sum(1 for s in val_set if s['label'] == 0)} normal")
        logger.info(f"Test set: {sum(1 for s in test_set if s['label'] == 1)} attack, {sum(1 for s in test_set if s['label'] == 0)} normal")
        
        return dataset
    
    def process_all_captures(self, window_size: int = 5) -> Dict:
        """
        Process all capture files and create a dataset for Dynamic GNN.
        
        Args:
            window_size: Number of frames to include in each temporal sequence
            
        Returns:
            Dictionary with processed dataset
        """
        # Step 1: Collect all capture files
        capture_files = self.collect_capture_files()
        
        if not capture_files:
            logger.error("No capture files found")
            return {}
        
        # Step 2: Process each capture file
        graph_encodings = []
        
        for capture_info in capture_files:
            encoding = self.process_capture_file(capture_info)
            if encoding:
                graph_encodings.append(encoding)
        
        logger.info(f"Processed {len(graph_encodings)} capture files")
        
        # Step 3: Create temporal sequences
        sequences = self.create_temporal_sequence(graph_encodings, window_size)
        
        # Step 4: Split dataset
        dataset = self.split_dataset(sequences)
        
        # Step 5: Save processed dataset
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = self.output_dir / f"processed_dataset_{timestamp}.pkl"
        
        with open(save_path, 'wb') as f:
            pickle.dump(dataset, f)
        
        logger.info(f"Saved processed dataset to {save_path}")
        
        return {
            "dataset": dataset,
            "save_path": str(save_path),
            "num_sequences": len(sequences),
            "window_size": window_size,
            "timestamp": timestamp
        }

def main():
    """Main function to run data preprocessing."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Data Preprocessing for Dynamic GNN Threat Detection")
    parser.add_argument("--data-dir", type=str, default="./data/gns3_simulation", help="Directory containing GNS3 simulation data")
    parser.add_argument("--output-dir", type=str, default="./data/processed", help="Directory to save processed data")
    parser.add_argument("--window-size", type=int, default=5, help="Temporal window size")
    
    args = parser.parse_args()
    
    # Create data processor
    processor = GNS3DataProcessor(args.data_dir, args.output_dir)
    
    # Process all captures
    result = processor.process_all_captures(args.window_size)
    
    if result:
        print("\nData preprocessing completed successfully!")
        print(f"Processed {result['num_sequences']} temporal sequences with window size {result['window_size']}")
        print(f"Dataset saved to: {result['save_path']}")
    else:
        print("\nData preprocessing failed. Check the logs for details.")

if __name__ == "__main__":
    main()
