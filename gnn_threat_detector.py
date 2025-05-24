#!/usr/bin/env python3
"""
GNN Threat Detector Integration
-------------------------------
This script integrates the trained Dynamic GNN model with the GNS3 monitoring
system for real-time network threat detection.

Usage:
    python gnn_threat_detector.py --model [path] [--monitor]
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import pickle
import time
import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any

# Try to import PyTorch and related libraries
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    print("Warning: PyTorch not installed. Model inference will not be available.")
    HAS_TORCH = False

# Try to import from main system modules
try:
    from dynamic_gnn_threat_detection import (
        DataIngestion,
        GraphEncoder,
        DataPreprocessor
    )
    from monitor_gns3 import (
        MonitoringController,
        NetworkCapture,
        ThreatDetectionIntegration
    )
    from train_dynamic_gnn import DynamicGNN, EvolveGCN
    HAS_SYSTEM_MODULES = True
except ImportError:
    print("Warning: Some system modules could not be imported.")
    HAS_SYSTEM_MODULES = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("gnn_threat_detector.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("gnn_threat_detector")

class GNNThreatDetector:
    """
    Main class for GNN-based network threat detection.
    
    This class loads a trained Dynamic GNN model and uses it for threat detection
    on network traffic data from GNS3 simulations or captures.
    """
    
    def __init__(self, model_path: str, config: Dict = None):
        """
        Initialize the GNN threat detector.
        
        Args:
            model_path: Path to the trained model checkpoint
            config: Configuration dictionary
        """
        self.model_path = model_path
        
        # Default configuration
        self.config = {
            "use_gpu": torch.cuda.is_available() if HAS_TORCH else False,
            "threshold": 0.7,  # Probability threshold for threat detection
            "window_size": 5,  # Number of frames in temporal sequence
            "data_dir": "./data/detector",
            "monitor_interval": 60,  # seconds
            "monitor_duration": 30,  # seconds
        }
        
        # Update with provided config
        if config:
            self.config.update(config)
        
        # Create data directory
        os.makedirs(self.config["data_dir"], exist_ok=True)
        
        # Initialize components if available
        if HAS_SYSTEM_MODULES:
            self.data_ingestion = DataIngestion()
            self.graph_encoder = GraphEncoder()
            self.data_preprocessor = DataPreprocessor()
        
        # Initialize model
        self.model = None
        self.device = torch.device('cuda' if self.config['use_gpu'] and HAS_TORCH and torch.cuda.is_available() else 'cpu')
        
        # Load model
        self._load_model()
        
        # Store recent captures for temporal processing
        self.recent_captures = []
    
    def _load_model(self):
        """Load the trained model from checkpoint."""
        if not HAS_TORCH:
            logger.error("PyTorch is required for model loading")
            return
        
        try:
            logger.info(f"Loading model from {self.model_path}")
            
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Get model type
            model_name = checkpoint.get('model_name', 'DynamicGNN')
            
            # Get model configuration
            model_config = checkpoint.get('config', {})
            hidden_channels = model_config.get('hidden_channels', 64)
            num_layers = model_config.get('num_layers', 3)
            dropout = model_config.get('dropout', 0.2)
            
            # Determine input dimension from model state dict
            # This assumes the first layer is named 'feature_transform.weight'
            state_dict = checkpoint['model_state_dict']
            input_dim = None
            if 'feature_transform.weight' in state_dict:
                input_dim = state_dict['feature_transform.weight'].shape[1]
            
            if input_dim is None:
                logger.error("Could not determine input dimension from model state dict")
                input_dim = 8  # Default fallback value
            
            # Create model based on type
            if model_name == 'EvolveGCN':
                self.model = EvolveGCN(
                    input_dim=input_dim,
                    hidden_channels=hidden_channels,
                    num_classes=2,
                    num_layers=num_layers,
                    dropout=dropout
                )
            else:  # Default to DynamicGNN
                self.model = DynamicGNN(
                    input_dim=input_dim,
                    hidden_channels=hidden_channels,
                    num_classes=2,
                    num_layers=num_layers,
                    dropout=dropout
                )
            
            # Load state dict
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Move to device and set to evaluation mode
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Model loaded successfully: {model_name} with {input_dim} input features")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            self.model = None
    
    def process_capture(self, capture_file: str) -> Dict:
        """
        Process a capture file for threat detection.
        
        Args:
            capture_file: Path to capture file
            
        Returns:
            Dictionary with detection results
        """
        if not HAS_TORCH or not HAS_SYSTEM_MODULES or self.model is None:
            logger.error("Model or required modules not available")
            return {"error": "Model or required modules not available"}
        
        try:
            logger.info(f"Processing capture file: {capture_file}")
            
            # Process capture file
            flow_df = self.data_ingestion.ingest_network_flow(capture_file)
            graph_encoding = self.graph_encoder.encode_network_data(flow_df)
            processed_encoding = self.data_preprocessor.preprocess_graph_encoding(graph_encoding)
            
            # Add metadata
            processed_encoding["metadata"] = {
                "filename": os.path.basename(capture_file),
                "timestamp": datetime.datetime.now().isoformat(),
                "filepath": capture_file
            }
            
            # Add to recent captures
            self.recent_captures.append(processed_encoding)
            
            # Keep only the most recent captures
            window_size = self.config["window_size"]
            if len(self.recent_captures) > window_size:
                self.recent_captures = self.recent_captures[-window_size:]
            
            # If we don't have enough captures yet, return no threat
            if len(self.recent_captures) < window_size:
                logger.info(f"Not enough captures for temporal analysis yet ({len(self.recent_captures)}/{window_size})")
                return {
                    "predicted_class": 0,
                    "probabilities": [1.0, 0.0],
                    "is_threat": False,
                    "message": f"Insufficient data for analysis ({len(self.recent_captures)}/{window_size} captures)",
                    "timestamp": datetime.datetime.now().isoformat()
                }
            
            # Create temporal sequence
            sequence = self.recent_captures[-window_size:]
            
            # Prepare input for model
            x_sequence = []
            edge_index_sequence = []
            
            for graph in sequence:
                # Convert node features to tensor
                x = torch.tensor(graph['node_features'], dtype=torch.float).to(self.device)
                x_sequence.append(x)
                
                # Convert edge index to tensor
                edge_index = torch.tensor(graph['edge_index'], dtype=torch.long).to(self.device)
                edge_index_sequence.append(edge_index)
            
            # Model inference
            self.model.eval()
            with torch.no_grad():
                # Forward pass
                outputs = self.model.forward_temporal(
                    [x_sequence], [edge_index_sequence], None)
                
                # Get predicted class and probabilities
                probabilities = F.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                probs = probabilities.cpu().numpy()[0].tolist()
            
            # Check if this is a threat based on threshold
            is_threat = predicted_class == 1 or probs[1] >= self.config["threshold"]
            
            result = {
                "predicted_class": predicted_class,
                "probabilities": probs,
                "is_threat": is_threat,
                "threat_probability": probs[1],
                "threshold": self.config["threshold"],
                "message": "Threat detected!" if is_threat else "No threat detected",
                "timestamp": datetime.datetime.now().isoformat(),
                "capture_file": capture_file
            }
            
            logger.info(f"Detection result: {result['message']} (prob: {probs[1]:.4f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing capture: {str(e)}")
            return {"error": str(e)}
    
    def start_monitoring(self):
        """Start monitoring network traffic for threats."""
        if not HAS_SYSTEM_MODULES:
            logger.error("System modules not available for monitoring")
            return {"error": "System modules not available"}
        
        try:
            logger.info("Starting network monitoring with GNN threat detection")
            
            # Create monitoring controller
            monitor_config = {
                "use_simulated_data": True,  # Change to False for real GNS3 data
                "capture_interval": self.config["monitor_interval"],
                "capture_duration": self.config["monitor_duration"],
                "data_dir": self.config["data_dir"],
                "alert_threshold": self.config["threshold"]
            }
            
            controller = MonitoringController(monitor_config)
            
            # Start monitoring
            controller.start_monitoring()
            
            # Run until interrupted
            try:
                while True:
                    # Get latest results
                    results = controller.get_latest_results(limit=5)
                    
                    if results:
                        for result in results:
                            # Process with our GNN model
                            capture_info = result.get("capture_info", {})
                            capture_file = capture_info.get("filepath")
                            
                            if capture_file and os.path.exists(capture_file):
                                # Process with GNN
                                detection_result = self.process_capture(capture_file)
                                
                                # Log result
                                if detection_result.get("is_threat", False):
                                    logger.warning(f"THREAT DETECTED in {os.path.basename(capture_file)}: "
                                                 f"Probability: {detection_result.get('threat_probability', 0):.4f}")
                                else:
                                    logger.info(f"No threat detected in {os.path.basename(capture_file)}")
                    
                    # Sleep before checking again
                    time.sleep(10)
                    
            except KeyboardInterrupt:
                logger.info("Monitoring interrupted by user")
            finally:
                # Stop monitoring
                controller.stop_monitoring()
                logger.info("Monitoring stopped")
            
            return {"status": "stopped", "message": "Monitoring stopped"}
            
        except Exception as e:
            logger.error(f"Error in monitoring: {str(e)}")
            return {"error": str(e)}
    
    def process_directory(self, directory: str) -> Dict:
        """
        Process all capture files in a directory.
        
        Args:
            directory: Directory containing capture files
            
        Returns:
            Dictionary with detection results for all files
        """
        if not os.path.isdir(directory):
            logger.error(f"Directory not found: {directory}")
            return {"error": f"Directory not found: {directory}"}
        
        # Find capture files
        capture_files = []
        for ext in ['.pcap', '.cap', '.pcapng']:
            capture_files.extend(list(Path(directory).glob(f"**/*{ext}")))
        
        logger.info(f"Found {len(capture_files)} capture files in {directory}")
        
        # Process each file
        results = []
        threats = []
        
        for capture_file in sorted(capture_files):
            result = self.process_capture(str(capture_file))
            results.append(result)
            
            if result.get("is_threat", False):
                threats.append({
                    "file": str(capture_file),
                    "probability": result.get("threat_probability", 0),
                    "timestamp": result.get("timestamp")
                })
        
        # Return summary
        return {
            "total_files": len(capture_files),
            "threats_detected": len(threats),
            "threat_files": threats,
            "results": results
        }


def main():
    """Main function for the GNN threat detector."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="GNN Threat Detector Integration")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--monitor", action="store_true", help="Start continuous monitoring")
    parser.add_argument("--process", type=str, help="Process a capture file or directory")
    parser.add_argument("--threshold", type=float, default=0.7, help="Threat detection threshold")
    parser.add_argument("--window-size", type=int, default=5, help="Temporal window size")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU usage")
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        logger.error(f"Model not found: {args.model}")
        return
    
    # Create configuration
    config = {
        "threshold": args.threshold,
        "window_size": args.window_size,
        "use_gpu": not args.no_gpu and torch.cuda.is_available() if HAS_TORCH else False
    }
    
    # Create detector
    detector = GNNThreatDetector(args.model, config)
    
    # Run in specified mode
    if args.monitor:
        # Start monitoring
        logger.info("Starting continuous monitoring")
        detector.start_monitoring()
    elif args.process:
        # Process file or directory
        path = args.process
        
        if os.path.isdir(path):
            # Process directory
            logger.info(f"Processing directory: {path}")
            results = detector.process_directory(path)
            
            # Print summary
            print(f"\nProcessed {results['total_files']} capture files")
            print(f"Detected {results['threats_detected']} potential threats")
            
            if results['threats_detected'] > 0:
                print("\nThreat detections:")
                for i, threat in enumerate(results['threat_files'], 1):
                    print(f"  {i}. {os.path.basename(threat['file'])}: "
                          f"Probability {threat['probability']:.4f}")
        else:
            # Process single file
            logger.info(f"Processing file: {path}")
            result = detector.process_capture(path)
            
            # Print result
            if "error" in result:
                print(f"Error: {result['error']}")
            else:
                print("\nDetection result:")
                print(f"  {'THREAT DETECTED!' if result['is_threat'] else 'No threat detected'}")
                print(f"  Threat probability: {result['probabilities'][1]:.4f}")
                print(f"  Threshold: {result['threshold']}")
    else:
        print("Error: Please specify --monitor or --process")


if __name__ == "__main__":
    if not HAS_TORCH:
        print("Error: PyTorch is required for model inference.")
        print("Please install with: pip install torch torch_geometric")
        sys.exit(1)
    
    main()
