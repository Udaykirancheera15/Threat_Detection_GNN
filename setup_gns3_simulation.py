#!/usr/bin/env python3
"""
GNS3 Simulation Setup for Dynamic GNN Threat Detection
------------------------------------------------------
This script sets up and runs GNS3 simulations to generate synthetic network
traffic data for training and testing a Dynamic GNN threat detection model.

Usage:
    python setup_gns3_simulation.py [--simulate] [--duration SECONDS]
"""

import os
import sys
import time
import argparse
import subprocess
import logging
from pathlib import Path

# Try to import GNS3 integration modules
try:
    from monitor_gns3 import MonitoringController, DEFAULT_CONFIG
    HAS_GNS3_MONITOR = True
except ImportError:
    print("Warning: GNS3 monitor module not found. Using simplified simulation.")
    HAS_GNS3_MONITOR = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("gns3_simulation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("gns3_simulation")

def setup_gns3_project(project_name="GNN_Threat_Detection"):
    """
    Set up a new GNS3 project for the threat detection simulation.
    
    Args:
        project_name: Name of the GNS3 project to create
        
    Returns:
        bool: True if setup was successful, False otherwise
    """
    logger.info(f"Setting up GNS3 project: {project_name}")
    
    if not HAS_GNS3_MONITOR:
        logger.warning("GNS3 monitor module not available. Using simulated setup.")
        return True
    
    # Create configuration for the monitoring controller
    config = DEFAULT_CONFIG.copy()
    config['project_name'] = project_name
    config['use_simulated_data'] = False
    config['data_dir'] = './data/gns3_simulation'
    
    # Initialize monitoring controller
    controller = MonitoringController(config)
    
    # Check if project already exists
    available_projects = controller.get_available_projects()
    if project_name in available_projects:
        logger.info(f"Project '{project_name}' already exists.")
        controller.select_project(project_name)
    else:
        logger.warning(f"Project '{project_name}' not found. Please create it manually in GNS3.")
        logger.info("Available projects: " + ", ".join(available_projects))
        return False
    
    return True

def generate_network_traffic(duration=600, use_simulation=False):
    """
    Generate network traffic data through GNS3 simulations.
    
    Args:
        duration: Duration of the simulation in seconds
        use_simulation: Whether to use simulated data (True) or actual GNS3 data (False)
        
    Returns:
        dict: Information about the generated traffic data
    """
    logger.info(f"Generating {'simulated' if use_simulation else 'GNS3'} network traffic for {duration} seconds")
    
    if not HAS_GNS3_MONITOR:
        # Fallback to calling the monitor script directly
        cmd = [
            sys.executable,
            "monitor_gns3.py",
            "--interval", "60",
            "--duration", "30",
            "--simulate" if use_simulation else "",
            "--data-dir", "./data/gns3_simulation"
        ]
        
        # Remove empty strings from command
        cmd = [c for c in cmd if c]
        
        try:
            logger.info(f"Running command: {' '.join(cmd)}")
            process = subprocess.Popen(cmd)
            
            # Wait for specified duration
            time.sleep(duration)
            
            # Terminate the process
            process.terminate()
            process.wait()
            
            logger.info("Network traffic generation completed")
            return {
                "success": True,
                "duration": duration,
                "simulated": use_simulation,
                "data_path": "./data/gns3_simulation"
            }
        except Exception as e:
            logger.error(f"Error running GNS3 monitor: {str(e)}")
            return {"success": False, "error": str(e)}
    else:
        # Use the monitoring controller directly
        config = DEFAULT_CONFIG.copy()
        config['use_simulated_data'] = use_simulation
        config['capture_interval'] = 60
        config['capture_duration'] = 30
        config['data_dir'] = './data/gns3_simulation'
        
        # Initialize monitoring controller
        controller = MonitoringController(config)
        
        try:
            # Start monitoring
            controller.start_monitoring()
            
            # Wait for specified duration
            logger.info(f"Collecting network traffic for {duration} seconds...")
            time.sleep(duration)
            
            # Stop monitoring
            controller.stop_monitoring()
            
            # Get captured results
            results = controller.get_latest_results(limit=100)
            
            logger.info(f"Network traffic generation completed with {len(results)} capture results")
            return {
                "success": True,
                "duration": duration,
                "simulated": use_simulation,
                "data_path": config['data_dir'],
                "num_captures": len(results)
            }
        except Exception as e:
            logger.error(f"Error in traffic generation: {str(e)}")
            return {"success": False, "error": str(e)}

def simulate_attacks(attack_types=None, duration=300, use_simulation=False):
    """
    Simulate different types of network attacks.
    
    Args:
        attack_types: List of attack types to simulate (default: all)
        duration: Duration of each attack simulation in seconds
        use_simulation: Whether to use simulated data
        
    Returns:
        dict: Information about the simulated attacks
    """
    if attack_types is None:
        attack_types = ["scan", "dos", "exploit"]
    
    logger.info(f"Simulating attacks: {', '.join(attack_types)}")
    
    if not HAS_GNS3_MONITOR:
        logger.warning("GNS3 monitor module not available. Using simplified attack simulation.")
        
        # Create data directory
        data_dir = Path("./data/gns3_simulation/attacks")
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Simulate each attack type
        for attack_type in attack_types:
            logger.info(f"Simulating {attack_type} attack...")
            
            # Create a marker file for the attack
            with open(data_dir / f"{attack_type}_attack.txt", 'w') as f:
                f.write(f"Simulated {attack_type} attack\n")
                f.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Duration: {duration} seconds\n")
            
            # Sleep to simulate duration
            time.sleep(5)  # Just a short delay for demonstration
        
        return {
            "success": True,
            "attack_types": attack_types,
            "simulated": True,
            "data_path": str(data_dir)
        }
    else:
        # Use the GNS3 integration from monitor_gns3.py
        from dynamic_gnn_threat_detection import GNS3Integration
        
        # Create GNS3 integration
        gns3_integration = GNS3Integration()
        
        # Try to connect to GNS3
        connected = gns3_integration.connect_to_gns3()
        
        if not connected and not use_simulation:
            logger.warning("Could not connect to GNS3. Falling back to simulation mode.")
            use_simulation = True
        
        attack_results = []
        
        # Simulate each attack type
        for attack_type in attack_types:
            logger.info(f"Simulating {attack_type} attack...")
            
            # Simulate the attack
            result = gns3_integration.simulate_attack(attack_type)
            
            # Collect network data during the attack
            data = gns3_integration.collect_and_process_data()
            
            attack_results.append({
                "attack_type": attack_type,
                "success": result,
                "data": data
            })
            
            # Sleep to allow separation between attack types
            time.sleep(30)
        
        return {
            "success": True,
            "attack_results": attack_results,
            "simulated": use_simulation
        }

def main():
    """Main function to run the GNS3 simulation setup."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="GNS3 Simulation Setup for Dynamic GNN Threat Detection")
    parser.add_argument("--simulate", action="store_true", help="Use simulated data instead of actual GNS3")
    parser.add_argument("--duration", type=int, default=600, help="Duration of normal traffic simulation in seconds")
    parser.add_argument("--attack-duration", type=int, default=300, help="Duration of each attack simulation in seconds")
    parser.add_argument("--project", type=str, default="GNN_Threat_Detection", help="GNS3 project name")
    
    args = parser.parse_args()
    
    # Create data directories
    data_dir = Path("./data/gns3_simulation")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Set up GNS3 project (skip if using simulation)
    if not args.simulate:
        setup_success = setup_gns3_project(args.project)
        if not setup_success:
            logger.warning("GNS3 project setup failed. Continuing with simulation mode.")
            args.simulate = True
    
    # Step 2: Generate normal network traffic
    logger.info("Generating normal network traffic...")
    normal_traffic = generate_network_traffic(args.duration, args.simulate)
    
    if not normal_traffic.get("success", False):
        logger.error("Failed to generate normal network traffic. Exiting.")
        return
    
    # Step 3: Simulate different attack types
    logger.info("Simulating network attacks...")
    attack_types = ["scan", "dos", "exploit"]
    attack_results = simulate_attacks(attack_types, args.attack_duration, args.simulate)
    
    # Step 4: Summarize the generated data
    logger.info("GNS3 simulation completed successfully.")
    logger.info(f"Normal traffic data: {normal_traffic}")
    logger.info(f"Attack simulation results: {attack_results}")
    logger.info(f"All data saved to: {data_dir}")
    
    print("\nGNS3 simulation completed successfully!")
    print(f"Generated data saved to: {data_dir}")
    print("Use this data to train your Dynamic GNN threat detection model.")

if __name__ == "__main__":
    main()
