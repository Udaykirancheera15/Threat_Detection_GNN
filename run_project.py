#!/usr/bin/env python3
"""
Dynamic GNN Threat Detection - Complete Project Workflow
-------------------------------------------------------
This script provides a complete workflow for the Dynamic GNN Threat Detection
project, including data generation, preprocessing, model training, and evaluation.

Usage:
    python run_project.py [--mode MODE] [--simulate]
"""

import os
import sys
import time
import subprocess
import logging
import argparse
import json
from pathlib import Path
import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("project_workflow.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("project_workflow")

def check_prerequisites():
    """Check if all required modules are installed."""
    missing_modules = []
    
    # Check core Python modules
    try:
        import numpy
        import pandas
    except ImportError as e:
        missing_modules.append(str(e).split("'")[1])
    
    # Check PyTorch
    try:
        import torch
        from torch import nn
    except ImportError:
        missing_modules.append("torch")
    
    # Check PyTorch Geometric
    try:
        import torch_geometric
    except ImportError:
        missing_modules.append("torch_geometric")
    
    if missing_modules:
        logger.warning(f"Missing required modules: {', '.join(missing_modules)}")
        return False
    
    return True

def check_scripts():
    """Check if all required scripts are present."""
    required_scripts = [
        "setup_gns3_simulation.py",
        "preprocess_data.py",
        "train_dynamic_gnn.py",
        "gnn_threat_detector.py"
    ]
    
    missing_scripts = []
    
    for script in required_scripts:
        if not os.path.exists(script):
            missing_scripts.append(script)
    
    if missing_scripts:
        logger.warning(f"Missing required scripts: {', '.join(missing_scripts)}")
        return False
    
    return True

def create_directories():
    """Create necessary directories for the project."""
    directories = [
        "data",
        "data/gns3_simulation",
        "data/processed",
        "models",
        "models/checkpoints",
        "results"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    logger.info(f"Created {len(directories)} directories")

def run_script(script_name, args=None, capture_output=False):
    """
    Run a Python script with the specified arguments.
    
    Args:
        script_name: Name of the script to run
        args: List of command-line arguments
        capture_output: Whether to capture and return the script output
        
    Returns:
        The completed process info, or None on failure
    """
    if args is None:
        args = []
    
    cmd = [sys.executable, script_name] + args
    logger.info(f"Running: {' '.join(cmd)}")
    
    try:
        if capture_output:
            process = subprocess.run(cmd, check=True, capture_output=True, text=True)
            return process
        else:
            process = subprocess.run(cmd, check=True)
            return process
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running {script_name}: {e}")
        return None

def generate_simulation_data(simulate=True):
    """
    Generate synthetic network traffic data.
    
    Args:
        simulate: Whether to use simulation (True) or actual GNS3 (False)
        
    Returns:
        Dictionary with information about the generated data
    """
    logger.info(f"Generating {'simulated' if simulate else 'GNS3'} network traffic data")
    
    # Set up arguments
    args = ["--duration", "600", "--attack-duration", "300"]
    if simulate:
        args.append("--simulate")
    
    # Run simulation script
    process = run_script("setup_gns3_simulation.py", args, capture_output=True)
    
    if process is None:
        logger.error("Failed to generate simulation data")
        return {"success": False}
    
    # Extract information from output
    output = process.stdout
    data_path = None
    for line in output.splitlines():
        if "Generated data saved to:" in line:
            data_path = line.split("Generated data saved to:", 1)[1].strip()
    
    return {
        "success": True,
        "simulated": simulate,
        "data_path": data_path
    }

def preprocess_data(data_dir):
    """
    Preprocess the simulation data for Dynamic GNN training.
    
    Args:
        data_dir: Directory containing simulation data
        
    Returns:
        Dictionary with information about the processed data
    """
    logger.info(f"Preprocessing data from {data_dir}")
    
    # Set up arguments
    args = ["--data-dir", data_dir, "--window-size", "5"]
    
    # Run preprocessing script
    process = run_script("preprocess_data.py", args, capture_output=True)
    
    if process is None:
        logger.error("Failed to preprocess data")
        return {"success": False}
    
    # Extract information from output
    output = process.stdout
    dataset_path = None
    for line in output.splitlines():
        if "Dataset saved to:" in line:
            dataset_path = line.split("Dataset saved to:", 1)[1].strip()
    
    return {
        "success": True,
        "dataset_path": dataset_path
    }

def train_model(dataset_path):
    """
    Train the Dynamic GNN model.
    
    Args:
        dataset_path: Path to the preprocessed dataset
        
    Returns:
        Dictionary with information about the trained model
    """
    logger.info(f"Training Dynamic GNN model with dataset {dataset_path}")
    
    # Set up arguments
    args = ["--dataset", dataset_path, "--output-dir", "models",
           "--model-type", "DynamicGNN", "--hidden-channels", "64",
           "--num-layers", "3", "--epochs", "30", "--patience", "5"]
    
    # Run training script
    process = run_script("train_dynamic_gnn.py", args, capture_output=True)
    
    if process is None:
        logger.error("Failed to train model")
        return {"success": False}
    
    # Extract information from output
    output = process.stdout
    model_path = None
    test_accuracy = None
    test_f1 = None
    
    for line in output.splitlines():
        if "Results saved to:" in line:
            model_path = line.split("Results saved to:", 1)[1].strip()
        elif "Test accuracy:" in line:
            test_accuracy = line.split("Test accuracy:", 1)[1].strip()
        elif "Test F1 score:" in line:
            test_f1 = line.split("Test F1 score:", 1)[1].strip()
    
    # Find the best model checkpoint
    checkpoint_dir = os.path.join("models", "checkpoints")
    best_model = None
    
    if os.path.exists(checkpoint_dir):
        checkpoints = list(Path(checkpoint_dir).glob("best_*.pt"))
        if checkpoints:
            best_model = str(sorted(checkpoints, key=lambda x: x.stat().st_mtime)[-1])
    
    return {
        "success": True,
        "model_path": model_path,
        "best_model": best_model,
        "test_accuracy": test_accuracy,
        "test_f1": test_f1
    }

def evaluate_model(model_path, data_dir):
    """
    Evaluate the trained model on test data.
    
    Args:
        model_path: Path to the trained model
        data_dir: Directory containing test data
        
    Returns:
        Dictionary with evaluation results
    """
    logger.info(f"Evaluating model {model_path} on data in {data_dir}")
    
    # Set up arguments
    args = ["--model", model_path, "--process", data_dir, "--threshold", "0.7"]
    
    # Run evaluation script
    process = run_script("gnn_threat_detector.py", args, capture_output=True)
    
    if process is None:
        logger.error("Failed to evaluate model")
        return {"success": False}
    
    # Extract information from output
    output = process.stdout
    total_files = None
    threats_detected = None
    
    for line in output.splitlines():
        if "Processed" in line and "capture files" in line:
            parts = line.split()
            total_files = parts[1]
        elif "Detected" in line and "potential threats" in line:
            parts = line.split()
            threats_detected = parts[1]
    
    return {
        "success": True,
        "total_files": total_files,
        "threats_detected": threats_detected,
        "output": output
    }

def run_complete_workflow(simulate=True):
    """
    Run the complete project workflow.
    
    Args:
        simulate: Whether to use simulation (True) or actual GNS3 (False)
        
    Returns:
        Dictionary with workflow results
    """
    workflow_start = time.time()
    
    # Create a timestamp for this run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logger.info(f"Starting complete workflow run at {timestamp}")
    
    # Step 1: Generate simulation data
    logger.info("Step 1: Generating simulation data")
    simulation_result = generate_simulation_data(simulate)
    
    if not simulation_result.get("success", False):
        logger.error("Workflow failed at Step 1: Generating simulation data")
        return {
            "success": False,
            "step_failed": "generate_simulation_data",
            "timestamp": timestamp
        }
    
    data_dir = simulation_result.get("data_path", "data/gns3_simulation")
    logger.info(f"Generated data saved to: {data_dir}")
    
    # Step 2: Preprocess data
    logger.info("Step 2: Preprocessing data")
    preprocess_result = preprocess_data(data_dir)
    
    if not preprocess_result.get("success", False):
        logger.error("Workflow failed at Step 2: Preprocessing data")
        return {
            "success": False,
            "step_failed": "preprocess_data",
            "timestamp": timestamp,
            "simulation_result": simulation_result
        }
    
    dataset_path = preprocess_result.get("dataset_path")
    logger.info(f"Preprocessed dataset saved to: {dataset_path}")
    
    # Step 3: Train model
    logger.info("Step 3: Training model")
    training_result = train_model(dataset_path)
    
    if not training_result.get("success", False):
        logger.error("Workflow failed at Step 3: Training model")
        return {
            "success": False,
            "step_failed": "train_model",
            "timestamp": timestamp,
            "simulation_result": simulation_result,
            "preprocess_result": preprocess_result
        }
    
    model_path = training_result.get("best_model")
    logger.info(f"Trained model saved to: {model_path}")
    
    # Step 4: Evaluate model
    logger.info("Step 4: Evaluating model")
    evaluation_result = evaluate_model(model_path, data_dir)
    
    if not evaluation_result.get("success", False):
        logger.error("Workflow failed at Step 4: Evaluating model")
        return {
            "success": False,
            "step_failed": "evaluate_model",
            "timestamp": timestamp,
            "simulation_result": simulation_result,
            "preprocess_result": preprocess_result,
            "training_result": training_result
        }
    
    # Calculate total runtime
    workflow_end = time.time()
    runtime_seconds = workflow_end - workflow_start
    runtime_hours, remainder = divmod(runtime_seconds, 3600)
    runtime_minutes, runtime_seconds = divmod(remainder, 60)
    runtime_formatted = f"{int(runtime_hours)}h {int(runtime_minutes)}m {int(runtime_seconds)}s"
    
    # Compile final results
    workflow_result = {
        "success": True,
        "timestamp": timestamp,
        "runtime": runtime_formatted,
        "runtime_seconds": runtime_seconds,
        "simulation_result": simulation_result,
        "preprocess_result": preprocess_result,
        "training_result": training_result,
        "evaluation_result": evaluation_result,
        "simulated": simulate
    }
    
    # Save results to JSON file
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"workflow_results_{timestamp}.json")
    
    with open(results_file, 'w') as f:
        json.dump(workflow_result, f, indent=2)
    
    logger.info(f"Complete workflow finished in {runtime_formatted}")
    logger.info(f"Results saved to {results_file}")
    
    return workflow_result

def run_monitoring_mode(model_path, simulate=True):
    """
    Run the model in continuous monitoring mode.
    
    Args:
        model_path: Path to the trained model
        simulate: Whether to use simulation (True) or actual GNS3 (False)
        
    Returns:
        None
    """
    logger.info(f"Starting monitoring mode with model {model_path}")
    
    # Set up arguments
    args = ["--model", model_path, "--monitor"]
    if simulate:
        # For simulation, GNN threat detector will use simulated data
        # No need for additional args as it's the default in gnn_threat_detector.py
        pass
    
    # Run the detector in monitoring mode
    run_script("gnn_threat_detector.py", args)

def print_summary(result):
    """Print a summary of the workflow result."""
    if not result.get("success", False):
        print("\n❌ Workflow failed!")
        print(f"Failed at step: {result.get('step_failed', 'unknown')}")
        return
    
    print("\n✅ Workflow completed successfully!")
    print(f"Total runtime: {result.get('runtime', 'unknown')}")
    
    # Simulation
    simulation_result = result.get("simulation_result", {})
    data_path = simulation_result.get("data_path", "unknown")
    print(f"\nSimulation data: {data_path}")
    
    # Preprocessing
    preprocess_result = result.get("preprocess_result", {})
    dataset_path = preprocess_result.get("dataset_path", "unknown")
    print(f"Preprocessed dataset: {dataset_path}")
    
    # Training
    training_result = result.get("training_result", {})
    model_path = training_result.get("best_model", "unknown")
    test_accuracy = training_result.get("test_accuracy", "unknown")
    test_f1 = training_result.get("test_f1", "unknown")
    print(f"\nTrained model: {model_path}")
    print(f"Test accuracy: {test_accuracy}")
    print(f"Test F1 score: {test_f1}")
    
    # Evaluation
    evaluation_result = result.get("evaluation_result", {})
    total_files = evaluation_result.get("total_files", "unknown")
    threats_detected = evaluation_result.get("threats_detected", "unknown")
    print(f"\nEvaluated {total_files} capture files")
    print(f"Detected {threats_detected} potential threats")
    
    # Results location
    results_dir = "results"
    print(f"\nDetailed results saved to: {results_dir}")
    
    # Next steps
    print("\nNext steps:")
    print(f"1. Run model in monitoring mode: python gnn_threat_detector.py --model {model_path} --monitor")
    print(f"2. Process specific captures: python gnn_threat_detector.py --model {model_path} --process <capture_file_or_dir>")

def main():
    """Main function for the project workflow."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Dynamic GNN Threat Detection - Complete Project Workflow")
    parser.add_argument("--mode", type=str, default="workflow", choices=["workflow", "monitor"],
                       help="Mode to run (workflow or monitor)")
    parser.add_argument("--simulate", action="store_true", help="Use simulation instead of actual GNS3")
    parser.add_argument("--model", type=str, help="Path to trained model (required for monitor mode)")
    
    args = parser.parse_args()
    
    # Check prerequisites
    if not check_prerequisites():
        print("Error: Missing required Python modules. Please install them before continuing.")
        print("Run: pip install numpy pandas torch torch_geometric")
        return
    
    # Check scripts
    if not check_scripts():
        print("Error: Missing required Python scripts. Please ensure all scripts are in the current directory.")
        return
    
    # Create directories
    create_directories()
    
    # Run in specified mode
    if args.mode == "workflow":
        # Run complete workflow
        result = run_complete_workflow(args.simulate)
        print_summary(result)
        
        # If successful and user wants to continue to monitoring, do that
        if result.get("success", False):
            model_path = result.get("training_result", {}).get("best_model")
            if model_path:
                response = input("\nDo you want to start monitoring with this model? (y/n): ")
                if response.lower() in ["y", "yes"]:
                    run_monitoring_mode(model_path, args.simulate)
    
    elif args.mode == "monitor":
        # Check if model is provided
        if not args.model:
            print("Error: --model argument is required for monitor mode.")
            return
        
        # Check if model exists
        if not os.path.exists(args.model):
            print(f"Error: Model file not found: {args.model}")
            return
        
        # Run in monitoring mode
        run_monitoring_mode(args.model, args.simulate)


if __name__ == "__main__":
    main()
