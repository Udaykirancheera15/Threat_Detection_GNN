#!/usr/bin/env python3
"""
Dynamic GNN Threat Detection System
----------------------------------
A comprehensive implementation of a Dynamic Graph Neural Network (GNN) system
for proactive threat detection in evolving cybersecurity landscapes.

This single-file implementation includes all core components:
- Data ingestion and preprocessing
- Attack graph construction using MulVAL integration
- Dynamic GNN model implementation
- Training and inference pipelines
- Explainability module
- GNS3 integration for network simulation
- Simple API endpoints for integration

Usage:
  python dynamic_gnn_threat_detection.py --mode [api|train|predict|demo|gns3]
  
  Modes:
    - api: Start the API server
    - train: Train a model with provided datasets
    - predict: Run inference with a trained model
    - demo: Generate synthetic data and run through the pipeline
    - gns3: Run with GNS3 integration for network simulation
"""

import os
import sys
import json
import logging
import datetime
import math
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
from collections import defaultdict, deque
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
import time
import random
import pickle
import subprocess
from tqdm import tqdm
import re

# Deep learning imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool
    from torch_geometric.utils import degree
    HAS_TORCH = True
except ImportError:
    warnings.warn("PyTorch or PyTorch Geometric not installed. Model training and inference will not be available.")
    HAS_TORCH = False

# For API
try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, File, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel, Field
    import uvicorn
    HAS_FASTAPI = True
except ImportError:
    warnings.warn("FastAPI not installed. API server will not be available.")
    HAS_FASTAPI = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dynamic_gnn.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("dynamic_gnn")

# Global configuration
CONFIG = {
    "data_dir": "./data",
    "models_dir": "./models",
    "mulval_path": "./mulval",
    "use_gpu": HAS_TORCH and torch.cuda.is_available(),
    "model_config": {
        "hidden_channels": 64,
        "num_layers": 3,
        "dropout": 0.2,
        "learning_rate": 0.001,
        "weight_decay": 5e-4,
        "epochs": 50,
        "patience": 10,
        "architecture": "DynamicGNN",  # DynamicGNN, EvolveGCN, DySAT
    },
    "api_host": "0.0.0.0",
    "api_port": 8000,
    "gns3_config": {
        "server": "localhost",
        "port": 3080,
        "project_name": "threat_detection_lab",
        "capture_interfaces": ["eth0"],
        "capture_duration": 60,  # seconds
        "capture_interval": 300,  # seconds
    },
    "mulval_options": {
        "simulate": True  # Set to True to simulate MulVAL output for testing
    }
}

# Make sure necessary directories exist
for dir_path in [CONFIG["data_dir"], CONFIG["models_dir"], CONFIG["mulval_path"]]:
    Path(dir_path).mkdir(parents=True, exist_ok=True)

#######################
# DATA INGESTION LAYER
#######################

class DataIngestion:
    """Handles ingestion of different data sources for threat detection."""
    
    def __init__(self, config: Dict = None):
        self.config = config or CONFIG
        self.data_dir = Path(self.config["data_dir"])
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def ingest_network_flow(self, file_path: str) -> pd.DataFrame:
        """
        Ingest network flow data from PCAP or CSV files.
        
        Args:
            file_path: Path to the network flow data file
            
        Returns:
            Pandas DataFrame with processed network flow data
        """
        logger.info(f"Ingesting network flow data from {file_path}")
        
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.csv':
            # Direct CSV ingestion
            df = pd.read_csv(file_path)
            return self._process_network_flow_df(df)
        elif file_ext == '.pcap':
            # Convert PCAP to CSV using tshark (requires tshark to be installed)
            output_csv = str(self.data_dir / f"pcap_converted_{int(time.time())}.csv")
            try:
                # Check if we have tshark installed
                tshark_check = subprocess.run(
                    ['which', 'tshark'],
                    check=False, capture_output=True, text=True
                )
                
                if tshark_check.returncode != 0:
                    logger.warning("tshark not found. Generating simulated network flow data.")
                    # Generate simulated data instead
                    return self._generate_simulated_network_data(50)
                
                subprocess.run(
                    ['tshark', '-r', file_path, '-T', 'fields', 
                     '-e', 'frame.time', '-e', 'ip.src', '-e', 'ip.dst',
                     '-e', 'tcp.srcport', '-e', 'tcp.dstport', '-e', 'udp.srcport',
                     '-e', 'udp.dstport', '-e', 'ip.proto', '-e', 'frame.len',
                     '-E', 'header=y', '-E', 'separator=,', '-o', 'tcp.desegment_tcp_streams:TRUE',
                     '-o', 'ip.defragment:TRUE', '-o', 'udp.check_checksum:FALSE'],
                    check=True, capture_output=True, text=True
                )
                
                if os.path.exists(output_csv):
                    df = pd.read_csv(output_csv)
                    return self._process_network_flow_df(df)
                else:
                    # If conversion fails, generate simulated data
                    logger.warning("Failed to generate CSV. Using simulated data.")
                    return self._generate_simulated_network_data(50)
                    
            except subprocess.SubprocessError as e:
                logger.error(f"Failed to convert PCAP to CSV: {str(e)}")
                logger.info("Generating simulated network flow data instead.")
                return self._generate_simulated_network_data(50)
        else:
            raise ValueError(f"Unsupported file extension: {file_ext}")
    
    def _process_network_flow_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process network flow DataFrame into a standardized format."""
        # Rename columns for standardization
        standardized_df = df.copy()
        
        # Map columns to standard names if they exist
        column_mappings = {
            "src_ip": ["src", "source", "ip.src", "source_ip"],
            "dst_ip": ["dst", "destination", "ip.dst", "destination_ip"],
            "src_port": ["sport", "source_port", "tcp.srcport", "udp.srcport"],
            "dst_port": ["dport", "destination_port", "tcp.dstport", "udp.dstport"],
            "protocol": ["proto", "ip.proto", "protocol_id"],
            "timestamp": ["time", "frame.time", "ts"],
            "bytes": ["size", "frame.len", "length"]
        }
        
        # Apply mappings
        for target, possible_names in column_mappings.items():
            # If target already exists, skip
            if target in standardized_df.columns:
                continue
                
            for name in possible_names:
                if name in df.columns:
                    standardized_df[target] = df[name]
                    break
            
        # Ensure required columns exist
        required_cols = ["src_ip", "dst_ip"]
        missing = [col for col in required_cols if col not in standardized_df.columns]
        if missing:
            logger.warning(f"Missing required columns: {missing}. Generating simulated data.")
            return self._generate_simulated_network_data(50)
            
        # Add timestamp if not present
        if "timestamp" not in standardized_df.columns:
            standardized_df["timestamp"] = pd.Timestamp.now()
            
        # Convert IP addresses to categorical codes for graph processing
        for ip_col in ["src_ip", "dst_ip"]:
            standardized_df[f"{ip_col}_code"] = standardized_df[ip_col].astype('category').cat.codes
            
        return standardized_df
    
    def _generate_simulated_network_data(self, num_flows: int = 100) -> pd.DataFrame:
        """Generate simulated network flow data for testing."""
        logger.info(f"Generating {num_flows} simulated network flows")
        
        # Generate common IP addresses
        ip_pool = [
            "192.168.1.1", "192.168.1.2", "192.168.1.3", "192.168.1.4",
            "192.168.1.100", "192.168.1.101", "192.168.1.102",
            "10.0.0.1", "10.0.0.2", "10.0.0.3",
            "172.16.0.1", "172.16.0.2"
        ]
        
        # Generate flows
        flows = []
        now = datetime.datetime.now()
        
        for i in range(num_flows):
            # Randomly select IPs
            src_ip = random.choice(ip_pool)
            dst_ip = random.choice([ip for ip in ip_pool if ip != src_ip])
            
            # Random timestamp within the last hour
            timestamp = now - datetime.timedelta(minutes=random.randint(0, 60))
            
            flows.append({
                'src_ip': src_ip,
                'dst_ip': dst_ip,
                'src_port': random.randint(1024, 65535),
                'dst_port': random.choice([80, 443, 22, 53, 3389, 8080]),
                'protocol': random.choice([6, 17, 1]),  # TCP, UDP, ICMP
                'bytes': random.randint(64, 1500),
                'timestamp': timestamp
            })
        
        df = pd.DataFrame(flows)
        
        # Add categorical codes
        for ip_col in ["src_ip", "dst_ip"]:
            df[f"{ip_col}_code"] = df[ip_col].astype('category').cat.codes
        
        return df
    
    def ingest_system_logs(self, file_path: str = None) -> pd.DataFrame:
        """
        Ingest system log data from various formats.
        
        Args:
            file_path: Path to the system log file
            
        Returns:
            Pandas DataFrame with processed log data
        """
        if file_path is None:
            # Generate synthetic logs
            return self._generate_simulated_logs(50)
            
        logger.info(f"Ingesting system logs from {file_path}")
        
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.csv':
            return pd.read_csv(file_path)
        elif file_ext == '.json':
            return pd.read_json(file_path)
        elif file_ext in ['.log', '.txt']:
            try:
                # Simple log parser for common log formats
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                
                log_entries = []
                for line in lines:
                    try:
                        # Basic parsing assuming timestamp at beginning
                        parts = line.split()
                        if len(parts) >= 3:
                            timestamp = " ".join(parts[0:2])
                            message = " ".join(parts[2:])
                            log_entries.append({
                                "timestamp": timestamp,
                                "message": message
                            })
                    except Exception as e:
                        logger.warning(f"Failed to parse log line: {line}. Error: {str(e)}")
                
                return pd.DataFrame(log_entries)
            except Exception as e:
                logger.error(f"Failed to parse log file: {str(e)}")
                return self._generate_simulated_logs(50)
        else:
            logger.warning(f"Unsupported file extension for logs: {file_ext}")
            return self._generate_simulated_logs(50)
    
    def _generate_simulated_logs(self, num_entries: int = 100) -> pd.DataFrame:
        """Generate simulated log entries for testing."""
        logger.info(f"Generating {num_entries} simulated log entries")
        
        # Define log message templates
        log_templates = [
            "User {user} logged in from {ip}",
            "Failed login attempt for user {user} from {ip}",
            "Connection {status} from {ip}",
            "Service {service} {status}",
            "CPU usage at {percentage}%",
            "Memory usage at {percentage}%",
            "Port scan detected from {ip}",
            "Firewall blocked traffic from {ip} to {dest_ip}",
            "Unusual traffic pattern detected for {service}",
            "Security alert: {alert_type} detected"
        ]
        
        # User names and services
        users = ["admin", "user1", "root", "system", "guest"]
        services = ["nginx", "apache", "mysql", "ssh", "ftp", "dns"]
        statuses = ["started", "stopped", "restarted", "failed", "timeout", "completed"]
        ips = [
            "192.168.1.1", "192.168.1.2", "192.168.1.100",
            "10.0.0.1", "10.0.0.2", "172.16.0.1", "8.8.8.8"
        ]
        alert_types = ["malware", "intrusion", "brute force", "data exfiltration", "unauthorized access"]
        
        # Generate log entries
        logs = []
        now = datetime.datetime.now()
        
        for i in range(num_entries):
            template = random.choice(log_templates)
            
            # Generate parameters
            params = {
                "user": random.choice(users),
                "ip": random.choice(ips),
                "dest_ip": random.choice(ips),
                "service": random.choice(services),
                "status": random.choice(statuses),
                "percentage": random.randint(10, 95),
                "alert_type": random.choice(alert_types)
            }
            
            # Format the message
            message = template.format(**{k: params[k] for k in re.findall(r'{(\w+)}', template)})
            
            # Random timestamp within the last day
            timestamp = now - datetime.timedelta(hours=random.randint(0, 24))
            
            logs.append({
                "timestamp": timestamp,
                "message": message,
                "level": random.choice(["INFO", "WARNING", "ERROR", "CRITICAL"]),
                "source": random.choice(["system", "security", "application", "firewall"])
            })
        
        return pd.DataFrame(logs)
    
    def ingest_vulnerability_scans(self, file_path: str = None) -> pd.DataFrame:
        """
        Ingest vulnerability scan results from standard formats.
        
        Args:
            file_path: Path to the vulnerability scan file
            
        Returns:
            Pandas DataFrame with processed vulnerability data
        """
        if file_path is None:
            # Generate synthetic vulnerability data
            return self._generate_simulated_vulnerabilities(10)
            
        logger.info(f"Ingesting vulnerability scan from {file_path}")
        
        file_ext = Path(file_path).suffix.lower()
        
        try:
            if file_ext == '.csv':
                return pd.read_csv(file_path)
            elif file_ext == '.json':
                return pd.read_json(file_path)
            elif file_ext == '.xml':
                # In a real implementation, this would parse actual XML vulnerability scans
                logger.warning("XML parsing not fully implemented. Generating simulated data.")
                return self._generate_simulated_vulnerabilities(10)
            else:
                logger.warning(f"Unsupported file extension for vulnerability scan: {file_ext}")
                return self._generate_simulated_vulnerabilities(10)
        except Exception as e:
            logger.error(f"Failed to parse vulnerability scan: {str(e)}")
            return self._generate_simulated_vulnerabilities(10)
    
    def _generate_simulated_vulnerabilities(self, num_vulns: int = 20) -> pd.DataFrame:
        """Generate simulated vulnerability data for testing."""
        logger.info(f"Generating {num_vulns} simulated vulnerabilities")
        
        # IP addresses
        ips = [
            "192.168.1.1", "192.168.1.2", "192.168.1.3", "192.168.1.100",
            "192.168.1.101", "192.168.1.102", "10.0.0.1", "10.0.0.2"
        ]
        
        # Vulnerability templates
        vuln_templates = [
            "Remote Code Execution in {service}",
            "SQL Injection in {service} interface",
            "Cross-Site Scripting in {service} web page",
            "Default {service} credentials",
            "Outdated {service} version",
            "Unpatched {service} service",
            "Privilege Escalation in {service}",
            "Information Disclosure in {service}",
            "Denial of Service vulnerability in {service}",
            "Authentication Bypass in {service}"
        ]
        
        # Services
        services = ["Apache", "Nginx", "MySQL", "SSH", "FTP", "DNS", "SMTP", "SMB", "RDP", "VPN"]
        
        # Generate vulnerabilities
        vulnerabilities = []
        
        for i in range(num_vulns):
            service = random.choice(services)
            template = random.choice(vuln_templates)
            name = template.format(service=service)
            
            # Generate CVE ID
            year = random.randint(2018, 2023)
            number = random.randint(1000, 9999)
            cve = f"CVE-{year}-{number}"
            
            vulnerabilities.append({
                "ip": random.choice(ips),
                "name": name,
                "cve": cve,
                "severity": round(random.uniform(1.0, 10.0), 1),
                "service": service,
                "port": random.choice([21, 22, 25, 53, 80, 443, 445, 3389, 8080])
            })
        
        return pd.DataFrame(vulnerabilities)
    
    def ingest_network_configuration(self, file_path: str = None) -> Dict:
        """
        Ingest network configuration data from various formats.
        
        Args:
            file_path: Path to the network configuration file
            
        Returns:
            Dictionary with processed network configuration data
        """
        if file_path is None:
            # Generate synthetic network configuration
            return self._generate_simulated_network_config()
            
        logger.info(f"Ingesting network configuration from {file_path}")
        
        try:
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext == '.json':
                with open(file_path, 'r') as f:
                    return json.load(f)
            elif file_ext == '.yaml' or file_ext == '.yml':
                try:
                    import yaml
                    with open(file_path, 'r') as f:
                        return yaml.safe_load(f)
                except ImportError:
                    logger.error("PyYAML is required to parse YAML files. Install with 'pip install pyyaml'")
                    return self._generate_simulated_network_config()
            elif file_ext == '.csv':
                df = pd.read_csv(file_path)
                return df.to_dict(orient='records')
            else:
                logger.warning(f"Unsupported file extension for network configuration: {file_ext}")
                return self._generate_simulated_network_config()
        except Exception as e:
            logger.error(f"Failed to parse network configuration: {str(e)}")
            return self._generate_simulated_network_config()
    
    def _generate_simulated_network_config(self) -> Dict:
        """Generate simulated network configuration for testing."""
        logger.info("Generating simulated network configuration")
        
        # Create a simple network topology
        hosts = []
        
        # Router
        hosts.append({
            "ip": "192.168.1.1",
            "service": "routerService",
            "service_port": "sshProtocol",
            "user": "adminUser",
            "role": "networkAdmin",
            "connections": [
                {"target": "192.168.1.100", "port": "httpProtocol", "protocol": "tcp"},
                {"target": "192.168.1.101", "port": "sshProtocol", "protocol": "tcp"},
                {"target": "192.168.1.102", "port": "sqlProtocol", "protocol": "tcp"}
            ]
        })
        
        # Web server
        hosts.append({
            "ip": "192.168.1.100",
            "service": "webServer",
            "service_port": "httpProtocol",
            "user": "webUser",
            "role": "webServerRole",
            "connections": [
                {"target": "192.168.1.101", "port": "sshProtocol", "protocol": "tcp"},
                {"target": "192.168.1.102", "port": "sqlProtocol", "protocol": "tcp"}
            ]
        })
        
        # Database server
        hosts.append({
            "ip": "192.168.1.101",
            "service": "dbServer",
            "service_port": "sqlProtocol",
            "user": "dbUser",
            "role": "dbServerRole",
            "connections": [
                {"target": "192.168.1.102", "port": "fileProtocol", "protocol": "tcp"}
            ]
        })
        
        # File server
        hosts.append({
            "ip": "192.168.1.102",
            "service": "fileServer",
            "service_port": "fileProtocol",
            "user": "fileUser",
            "role": "fileServerRole",
            "connections": []
        })
        
        return {"hosts": hosts}
    
    def save_data(self, data: Union[pd.DataFrame, Dict], data_type: str, filename: str = None) -> str:
        """
        Save ingested data to the appropriate directory.
        
        Args:
            data: DataFrame or Dictionary to save
            data_type: Type of data ('network_flow', 'logs', 'vulnerabilities', 'config')
            filename: Optional filename, will be generated if not provided
            
        Returns:
            Path to the saved file
        """
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{data_type}_{timestamp}"
        
        # Create data type specific directory
        save_dir = self.data_dir / data_type
        save_dir.mkdir(exist_ok=True)
        
        if isinstance(data, pd.DataFrame):
            save_path = save_dir / f"{filename}.csv"
            data.to_csv(save_path, index=False)
        else:
            save_path = save_dir / f"{filename}.json"
            with open(save_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        logger.info(f"Saved {data_type} data to {save_path}")
        return str(save_path)


##########################
# GNS3 INTEGRATION LAYER
##########################

class GNS3Integration:
    """Integration with GNS3 for network simulation and data collection."""
    
    def __init__(self, config: Dict = None):
        self.config = config or CONFIG
        self.gns3_config = self.config.get("gns3_config", {})
        self.data_dir = Path(self.config["data_dir"]) / "gns3"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if GNS3 Python client is installed
        try:
            import gns3fy
            self.gns3fy = gns3fy
            self.has_gns3fy = True
            logger.info("Found GNS3fy module - full GNS3 integration available")
        except ImportError:
            logger.warning("GNS3fy module not found. Limited GNS3 integration available.")
            logger.warning("Install with: pip install gns3fy")
            self.has_gns3fy = False
    
    def connect_to_gns3(self):
        """Connect to GNS3 server."""
        if not self.has_gns3fy:
            logger.warning("GNS3fy module not available. Using simulated GNS3 mode.")
            return False
        
        try:
            # Connect to GNS3 server
            server_url = f"http://{self.gns3_config.get('server', 'localhost')}:{self.gns3_config.get('port', 3080)}"
            self.server = self.gns3fy.Gns3Connector(server_url)
            
            # Test connection
            try:
                projects = self.server.get_projects()
                logger.info(f"Connected to GNS3 server at {server_url}")
            except Exception as e:
                logger.error(f"Failed to connect to GNS3 server: {str(e)}")
                logger.warning("Using simulated GNS3 mode.")
                return False
            
            # Get project
            project_name = self.gns3_config.get("project_name", "threat_detection_lab")
            project_exists = any(p["name"] == project_name for p in projects)
            
            if not project_exists:
                logger.warning(f"Project '{project_name}' not found. Creating new project.")
                self.project = self.gns3fy.Project(name=project_name, connector=self.server)
                self.project.create()
            else:
                self.project = self.gns3fy.Project(name=project_name, connector=self.server)
                self.project.get()
            
            logger.info(f"Using GNS3 project: {project_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to GNS3 server: {str(e)}")
            logger.warning("Using simulated GNS3 mode.")
            return False
    
    def create_basic_topology(self):
        """Create a basic topology for threat detection simulation."""
        if not hasattr(self, 'project') or not self.has_gns3fy:
            logger.warning("Not connected to GNS3 or GNS3fy module not available.")
            logger.info("Creating simulated topology...")
            return False
        
        try:
            # Check if the project already has nodes
            self.project.get()
            if len(self.project.nodes) > 0:
                logger.info(f"Project already has {len(self.project.nodes)} nodes. Skipping topology creation.")
                return True
            
            logger.info("Creating basic network topology for threat simulation")
            
            # Look for available node templates
            templates = self.server.get_templates()
            available_types = {t["name"]: t for t in templates}
            
            # Try to find a router template
            router_template = None
            for key in ["Router", "Cisco IOSv", "Dynamips", "VyOS"]:
                if key in available_types:
                    router_template = available_types[key]
                    break
            
            # Try to find a Linux template
            linux_template = None
            for key in ["Ubuntu", "Linux", "Kali", "Debian", "Alpine"]:
                if key in available_types:
                    linux_template = available_types[key]
                    break
            
            if not router_template:
                logger.warning("No suitable router template found")
                return False
            
            if not linux_template:
                logger.warning("No suitable Linux template found")
                return False
            
            # Create router
            router = self.gns3fy.Node(
                project_id=self.project.project_id,
                name="Router",
                template=router_template["name"],
                compute_id="local",
                connector=self.server
            )
            router.create()
            
            # Create attacker machine
            attacker = self.gns3fy.Node(
                project_id=self.project.project_id,
                name="Attacker",
                template=linux_template["name"],
                compute_id="local",
                connector=self.server
            )
            attacker.create()
            
            # Create victim machine
            victim = self.gns3fy.Node(
                project_id=self.project.project_id,
                name="Victim",
                template=linux_template["name"],
                compute_id="local",
                connector=self.server
            )
            victim.create()
            
            # Wait for nodes to be created
            time.sleep(2)
            
            # Create links
            # Router to Attacker
            link1 = self.gns3fy.Link(
                project_id=self.project.project_id,
                connector=self.server,
                nodes=[
                    {"node_id": router.node_id, "adapter_number": 0, "port_number": 0},
                    {"node_id": attacker.node_id, "adapter_number": 0, "port_number": 0}
                ]
            )
            link1.create()
            
            # Router to Victim
            link2 = self.gns3fy.Link(
                project_id=self.project.project_id,
                connector=self.server,
                nodes=[
                    {"node_id": router.node_id, "adapter_number": 0, "port_number": 1},
                    {"node_id": victim.node_id, "adapter_number": 0, "port_number": 0}
                ]
            )
            link2.create()
            
            logger.info("Basic topology created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create topology: {str(e)}")
            return False
    
    def simulate_packet_capture(self):
        """
        Simulate packet capture when GNS3 is not available.
        
        Returns:
            List of paths to simulated capture files
        """
        logger.info("Simulating packet capture")
        
        # Create simulated packet capture files
        capture_files = []
        for i in range(2):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"simulated_capture_{i}_{timestamp}.pcap"
            capture_path = self.data_dir / filename
            
            # Create empty file as placeholder
            with open(capture_path, 'w') as f:
                f.write(f"# Simulated packet capture {i}\n")
                f.write(f"# Generated at {datetime.datetime.now().isoformat()}\n")
                f.write(f"# This is a placeholder file for GNS3 packet capture\n")
            
            capture_files.append(str(capture_path))
            logger.info(f"Created simulated capture file: {capture_path}")
        
        return capture_files
    
    def start_packet_capture(self):
        """
        Start packet capture on GNS3 links.
        
        Returns:
            List of paths to capture files or None on failure
        """
        if not hasattr(self, 'project') or not self.has_gns3fy:
            logger.warning("Not connected to GNS3 or GNS3fy module not available.")
            return self.simulate_packet_capture()
        
        try:
            # Find the nodes with specified interfaces
            captures = []
            self.project.get()
            
            for node in self.project.nodes:
                node_obj = self.gns3fy.Node(
                    project_id=self.project.project_id,
                    node_id=node["node_id"],
                    connector=self.server
                )
                node_obj.get()
                
                for link in node_obj.links:
                    # Start capture on links
                    capture_file = f"capture_{node['name']}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pcap"
                    capture = self.project.create_link_capture(link_id=link["link_id"], capture_file_name=capture_file)
                    captures.append((capture, capture_file))
                    logger.info(f"Started capture on link for {node['name']}, saved to {capture_file}")
            
            # Wait for the specified duration
            capture_duration = self.gns3_config.get("capture_duration", 60)
            logger.info(f"Capturing packets for {capture_duration} seconds...")
            time.sleep(capture_duration)
            
            # Stop captures
            capture_files = []
            for capture, filename in captures:
                self.project.stop_capture(capture_id=capture["capture_id"])
                logger.info(f"Stopped capture {filename}")
                
                # Download the capture file
                capture_path = self.data_dir / filename
                
                # In a real implementation, you would download the file from GNS3 server
                # For now, we'll create a placeholder
                with open(capture_path, 'w') as f:
                    f.write(f"# GNS3 packet capture: {filename}\n")
                    f.write(f"# Generated at {datetime.datetime.now().isoformat()}\n")
                    f.write(f"# This is a placeholder for actual PCAP data\n")
                
                capture_files.append(str(capture_path))
            
            logger.info(f"Packet capture completed. Saved {len(capture_files)} files.")
            return capture_files
            
        except Exception as e:
            logger.error(f"Failed to capture packets: {str(e)}")
            return self.simulate_packet_capture()
    
    def simulate_attack(self, attack_type: str = "scan"):
        """Simulate an attack in the GNS3 environment."""
        if not hasattr(self, 'project') or not self.has_gns3fy:
            logger.warning("Not connected to GNS3 or GNS3fy module not available.")
            logger.info(f"Simulating {attack_type} attack...")
            return False
        
        try:
            self.project.get()
            
            # Find attacker node
            attacker = None
            for node in self.project.nodes:
                if "attacker" in node["name"].lower():
                    attacker = self.gns3fy.Node(
                        project_id=self.project.project_id,
                        node_id=node["node_id"],
                        connector=self.server
                    )
                    attacker.get()
                    break
            
            if not attacker:
                logger.error("Attacker node not found in topology")
                return False
            
            # Find victim node
            victim = None
            for node in self.project.nodes:
                if "victim" in node["name"].lower():
                    victim = self.gns3fy.Node(
                        project_id=self.project.project_id,
                        node_id=node["node_id"],
                        connector=self.server
                    )
                    victim.get()
                    break
            
            if not victim:
                logger.error("Victim node not found in topology")
                return False
            
            # Determine command based on attack type
            command = ""
            if attack_type == "scan":
                # Nmap scan
                command = f"nmap -A {victim.console_host}"
            elif attack_type == "dos":
                # Simple DoS attack simulation
                command = f"ping -f {victim.console_host}"
            elif attack_type == "exploit":
                # Placeholder for an exploit (would be more complex in reality)
                command = f"echo 'Simulating exploit against {victim.console_host}'"
            
            # Run command on attacker node
            logger.info(f"Simulating {attack_type} attack from {attacker.name} to {victim.name}")
            logger.info(f"Command: {command}")
            
            # In a real implementation, you would execute the command via the GNS3 console
            # But here we're just simulating it
            logger.info("Attack simulation complete")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to simulate attack: {str(e)}")
            logger.info(f"Simulating {attack_type} attack...")
            return False
    
    def collect_and_process_data(self):
        """
        Collect and process network data from GNS3 simulation.
        
        Returns:
            Dictionary with collected data
        """
        logger.info("Collecting and processing data from GNS3")
        
        # Start packet capture
        capture_files = self.start_packet_capture()
        
        if not capture_files:
            logger.warning("No capture files generated")
            return None
        
        # Process the capture files
        data_ingestion = DataIngestion(self.config)
        
        # Process each capture file
        network_flows = []
        for capture_file in capture_files:
            try:
                flow_df = data_ingestion.ingest_network_flow(capture_file)
                network_flows.append(flow_df)
            except Exception as e:
                logger.error(f"Failed to process capture file {capture_file}: {str(e)}")
        
        # Combine all network flows
        if network_flows:
            combined_flows = pd.concat(network_flows, ignore_index=True)
            
            # Save combined flows
            flow_path = data_ingestion.save_data(combined_flows, "network_flow", "gns3_combined_flows")
            
            return {
                "network_flow": flow_path,
                "num_flows": len(combined_flows)
            }
        else:
            logger.warning("No valid network flows collected")
            return None
    
    def run_gns3_simulation(self):
        """
        Run full GNS3 simulation workflow.
        
        Returns:
            Dictionary with simulation results
        """
        logger.info("Starting GNS3 simulation workflow")
        
        # Step 1: Connect to GNS3
        connected = self.connect_to_gns3()
        
        # Step 2: Create topology if connected
        if connected:
            topology_created = self.create_basic_topology()
            if not topology_created:
                logger.warning("Failed to create topology. Using simulated data.")
        
        # Step 3: Simulate attack
        self.simulate_attack(attack_type="scan")
        
        # Step 4: Collect and process data
        data = self.collect_and_process_data()
        
        if data:
            logger.info(f"GNS3 simulation completed successfully. Collected {data.get('num_flows', 0)} network flows.")
            return data
        else:
            logger.warning("GNS3 simulation did not produce valid data")
            return None


#############################
# GRAPH CONSTRUCTION LAYER
#############################

class MulVALIntegration:
    """
    Interface for the MulVAL attack graph generator.
    
    MulVAL (Multi-host, Multi-stage Vulnerability Analysis Language) is a 
    practical tool for automatically identifying security vulnerabilities
    in network configurations.
    """
    
    def __init__(self, mulval_path: str = None, config: Dict = None):
        self.config = config or CONFIG
        self.mulval_path = mulval_path or self.config["mulval_path"]
        Path(self.mulval_path).mkdir(parents=True, exist_ok=True)
        
        # Check if MulVAL exists and is properly installed
        # This is a simple check - in a real implementation, more robust checks would be needed
        if not Path(self.mulval_path).exists():
            logger.warning(f"MulVAL path {self.mulval_path} does not exist. MulVAL functionality will be limited.")
    
    def prepare_mulval_input(self, 
                            vulnerability_data: pd.DataFrame, 
                            network_config: Dict) -> str:
        """
        Generate MulVAL input files from vulnerability and network data.
        
        Args:
            vulnerability_data: DataFrame with vulnerability information
            network_config: Network configuration as a dictionary
            
        Returns:
            Path to generated MulVAL input file
        """
        logger.info("Preparing MulVAL input file")
        
        # Create input directory if it doesn't exist
        input_dir = Path(self.mulval_path) / "inputs"
        input_dir.mkdir(exist_ok=True)
        
        # Generate a unique filename for this analysis
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        input_file = input_dir / f"mulval_input_{timestamp}.P"
        
        with open(input_file, 'w') as f:
            # Write header
            f.write("% MulVAL input file generated by Dynamic GNN Threat Detection\n")
            f.write(f"% Generated at: {datetime.datetime.now().isoformat()}\n\n")
            
            # Network topology
            f.write("% Network topology\n")
            for host in network_config.get("hosts", []):
                f.write(f"attackerLocated(internet).\n")
                f.write(f"hacl(internet, '{host['ip']}', {host.get('service_port', 'httpProtocol')}, {host.get('protocol', 'tcp')}).\n")
                
                # Add connections between hosts
                for connection in host.get("connections", []):
                    f.write(f"hacl('{host['ip']}', '{connection['target']}', {connection.get('port', 'httpProtocol')}, {connection.get('protocol', 'tcp')}).\n")
            
            f.write("\n% Host configurations\n")
            for host in network_config.get("hosts", []):
                f.write(f"networkServiceInfo('{host['ip']}', {host.get('service', 'webServer')}, {host.get('service_port', 'httpProtocol')}, {host.get('user', 'someUser')}, {host.get('role', 'serviceRole')}).\n")
            
            f.write("\n% Vulnerabilities\n")
            # Process vulnerability data and write to file
            for _, vuln in vulnerability_data.iterrows():
                # Extract vulnerability details
                ip = vuln.get('ip', 'unknown')
                cve = vuln.get('cve', 'CVE-unknown')
                name = vuln.get('name', 'unknown-vulnerability')
                severity = vuln.get('severity', '5.0')
                
                # Convert severity to integer if it's a string with a decimal point
                try:
                    if isinstance(severity, str) and '.' in severity:
                        severity = int(float(severity))
                except ValueError:
                    severity = 5  # Default medium severity
                
                # Write vulnerability to input file
                f.write(f"vulExists('{ip}', '{cve}', '{name}').\n")
                f.write(f"vulProperty('{cve}', remoteExploit, {severity}).\n")
                f.write(f"cvss('{cve}', {float(severity) if isinstance(severity, (int, float)) else 5.0}).\n")
            
            # In a complete implementation, add more Prolog rules based on system configuration
            f.write("\n% Additional rules\n")
            f.write("attackGoal(execCode(victim, root)).\n")
        
        logger.info(f"MulVAL input file created at {input_file}")
        return str(input_file)
    
    def run_mulval(self, input_file: str) -> Dict[str, str]:
        """
        Execute MulVAL to generate attack graph.
        
        Args:
            input_file: Path to MulVAL input file
            
        Returns:
            Dictionary with paths to output files
        """
        logger.info(f"Running MulVAL with input file {input_file}")
        
        # Create output directory if it doesn't exist
        output_dir = Path(self.mulval_path) / "outputs"
        output_dir.mkdir(exist_ok=True)
        
        # Generate output file paths
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base = output_dir / f"mulval_output_{timestamp}"
        output_files = {
            "attack_graph": f"{output_base}_AttackGraph.xml",
            "attack_graph_txt": f"{output_base}_AttackGraph.txt",
            "attack_paths": f"{output_base}_AttackPaths.txt",
            "visualization": f"{output_base}_graph.pdf",
        }
        
        try:
            # Check if we're in simulation mode (for testing without MulVAL installed)
            if "simulate" in self.config.get("mulval_options", {}) and self.config["mulval_options"]["simulate"]:
                logger.info("In MulVAL simulation mode")
                self._simulate_mulval_output(input_file, output_files)
            else:
                # Try to run actual MulVAL
                try:
                    # Check if MulVAL is installed
                    mulval_check = subprocess.run(
                        ['which', 'mulval'],
                        check=False, capture_output=True, text=True
                    )
                    
                    if mulval_check.returncode != 0:
                        logger.warning("MulVAL command not found. Using simulation mode.")
                        self._simulate_mulval_output(input_file, output_files)
                    else:
                        # Actual MulVAL command
                        cmd = [
                            'sh', 
                            f"{self.mulval_path}/utils/graph_gen.sh", 
                            input_file,
                            "-v",  # Verbose output
                            "-p",  # Generate attack paths
                            "-r"   # Render graph
                        ]
                        
                        process = subprocess.run(
                            cmd,
                            check=True,
                            capture_output=True,
                            text=True
                        )
                        
                        logger.info(f"MulVAL output: {process.stdout}")
                        if process.stderr:
                            logger.warning(f"MulVAL warnings/errors: {process.stderr}")
                except Exception as e:
                    logger.warning(f"Failed to run MulVAL: {str(e)}")
                    logger.info("Using MulVAL simulation mode instead")
                    self._simulate_mulval_output(input_file, output_files)
            
            return output_files
            
        except Exception as e:
            logger.error(f"Error in MulVAL processing: {str(e)}")
            raise RuntimeError(f"Error in MulVAL processing: {str(e)}")
    
    def _simulate_mulval_output(self, input_file: str, output_files: Dict[str, str]):
        """
        Generate simulated MulVAL output for testing.
        
        Args:
            input_file: Path to input file
            output_files: Dictionary with output file paths
        """
        # Read input file to extract nodes
        try:
            with open(input_file, 'r') as f:
                input_content = f.read()
                
            # Extract IPs
            import re
            ip_pattern = r"'((?:\d{1,3}\.){3}\d{1,3})'"
            ip_addresses = re.findall(ip_pattern, input_content)
            
            # Extract vulnerabilities
            cve_pattern = r"'(CVE-[^']+)'"
            cves = re.findall(cve_pattern, input_content)
            
            # Create a simple attack graph
            # In XML format (simplified)
            with open(output_files["attack_graph"], 'w') as f:
                f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
                f.write('<attack_graph>\n')
                
                # Add vertices (nodes)
                f.write('  <vertices>\n')
                node_id = 1
                for ip in ip_addresses:
                    f.write(f'    <vertex id="{node_id}">\n')
                    f.write(f'      <fact>hostAccess({ip})</fact>\n')
                    f.write(f'      <type>LEAF</type>\n')
                    f.write('    </vertex>\n')
                    node_id += 1
                
                for cve in cves:
                    f.write(f'    <vertex id="{node_id}">\n')
                    f.write(f'      <fact>vulExists(someHost, {cve}, someApp)</fact>\n')
                    f.write(f'      <type>LEAF</type>\n')
                    f.write('    </vertex>\n')
                    node_id += 1
                
                # Add AND nodes
                for i in range(min(len(ip_addresses), len(cves))):
                    f.write(f'    <vertex id="{node_id}">\n')
                    f.write(f'      <fact>AND({i+1}, {len(ip_addresses) + i + 1})</fact>\n')
                    f.write(f'      <type>AND</type>\n')
                    f.write('    </vertex>\n')
                    node_id += 1
                
                # Add a goal node
                f.write(f'    <vertex id="{node_id}">\n')
                f.write(f'      <fact>execCode(targetHost, root)</fact>\n')
                f.write(f'      <type>OR</type>\n')
                f.write('    </vertex>\n')
                f.write('  </vertices>\n')
                
                # Add arcs (edges)
                f.write('  <arcs>\n')
                # Connect hosts to AND nodes
                edge_id = 1
                and_node_start = len(ip_addresses) + len(cves) + 1
                for i, ip_idx in enumerate(range(1, len(ip_addresses) + 1)):
                    if i < min(len(ip_addresses), len(cves)):
                        f.write(f'    <arc id="{edge_id}" src="{ip_idx}" dst="{and_node_start + i}" />\n')
                        edge_id += 1
                
                # Connect vulnerabilities to AND nodes
                for i, cve_idx in enumerate(range(len(ip_addresses) + 1, len(ip_addresses) + len(cves) + 1)):
                    if i < min(len(ip_addresses), len(cves)):
                        f.write(f'    <arc id="{edge_id}" src="{cve_idx}" dst="{and_node_start + i}" />\n')
                        edge_id += 1
                
                # Connect AND nodes to goal
                goal_node = node_id
                for i in range(min(len(ip_addresses), len(cves))):
                    f.write(f'    <arc id="{edge_id}" src="{and_node_start + i}" dst="{goal_node}" />\n')
                    edge_id += 1
                
                f.write('  </arcs>\n')
                f.write('</attack_graph>\n')
            
            # Text format (simplified)
            with open(output_files["attack_graph_txt"], 'w') as f:
                f.write("MulVAL Attack Graph (Simulated)\n")
                f.write("==============================\n\n")
                f.write("Vertices:\n")
                node_id = 1
                for ip in ip_addresses:
                    f.write(f"{node_id}. hostAccess({ip}) LEAF\n")
                    node_id += 1
                for cve in cves:
                    f.write(f"{node_id}. vulExists(someHost, {cve}, someApp) LEAF\n")
                    node_id += 1
                
                and_node_start = node_id
                for i in range(min(len(ip_addresses), len(cves))):
                    src1 = i + 1
                    src2 = len(ip_addresses) + i + 1
                    f.write(f"{node_id}. AND({src1}, {src2}) AND\n")
                    node_id += 1
                
                f.write(f"{node_id}. execCode(targetHost, root) OR\n\n")
                
                f.write("Arcs:\n")
                edge_id = 1
                for i, ip_idx in enumerate(range(1, len(ip_addresses) + 1)):
                    if i < min(len(ip_addresses), len(cves)):
                        f.write(f"{edge_id}. {ip_idx} -> {and_node_start + i}\n")
                        edge_id += 1
                
                for i, cve_idx in enumerate(range(len(ip_addresses) + 1, len(ip_addresses) + len(cves) + 1)):
                    if i < min(len(ip_addresses), len(cves)):
                        f.write(f"{edge_id}. {cve_idx} -> {and_node_start + i}\n")
                        edge_id += 1
                
                goal_node = node_id
                for i in range(min(len(ip_addresses), len(cves))):
                    f.write(f"{edge_id}. {and_node_start + i} -> {goal_node}\n")
                    edge_id += 1
            
            # Attack paths
            with open(output_files["attack_paths"], 'w') as f:
                f.write("MulVAL Attack Paths (Simulated)\n")
                f.write("==============================\n\n")
                f.write("Shortest path to goal:\n")
                if ip_addresses and cves:
                    f.write(f"1. hostAccess({ip_addresses[0]})\n")
                    f.write(f"2. vulExists(someHost, {cves[0]}, someApp)\n")
                    f.write(f"3. AND(1, {len(ip_addresses) + 1})\n")
                    f.write(f"4. execCode(targetHost, root)\n")
            
            # For visualization, we would generate a PDF
            # Since we can't do that easily, we'll create a placeholder
            with open(output_files["visualization"], 'w') as f:
                f.write("Simulated MulVAL attack graph visualization\n")
                f.write("This would be a PDF in a real implementation.\n")
            
            logger.info("Simulated MulVAL output created")
            
        except Exception as e:
            logger.error(f"Error generating simulated MulVAL output: {str(e)}")
            # Create empty placeholder files
            for file_path in output_files.values():
                with open(file_path, 'w') as f:
                    f.write("Error: Failed to generate simulated output.\n")
    
    def parse_attack_graph(self, attack_graph_file: str) -> nx.DiGraph:
        """
        Parse MulVAL attack graph XML output into a NetworkX DiGraph.
        
        Args:
            attack_graph_file: Path to MulVAL attack graph XML file
            
        Returns:
            NetworkX DiGraph representing the attack graph
        """
        import re  # Add this line to import the re module
        logger.info(f"Parsing attack graph from {attack_graph_file}")
        
        import xml.etree.ElementTree as ET

        
        try:
            tree = ET.parse(attack_graph_file)
            root = tree.getroot()
            
            # Create directed graph
            G = nx.DiGraph()
            
            # Add vertices
            for vertex in root.findall('./vertices/vertex'):
                node_id = int(vertex.get('id'))
                fact = vertex.find('fact').text if vertex.find('fact') is not None else f"Node {node_id}"
                node_type = vertex.find('type').text if vertex.find('type') is not None else "UNKNOWN"
                
                # Add attributes to the node
                G.add_node(node_id, fact=fact, type=node_type)
                
                # Extract additional metadata for vulnerability nodes
                if 'vulExists' in fact:
                    # Extract CVE ID if present
                    import re
                    cve_match = re.search(r'CVE-\d+-\d+', fact)
                    if cve_match:
                        G.nodes[node_id]['cve'] = cve_match.group(0)
                
                # Extract host information for hostAccess nodes
                if 'hostAccess' in fact:
                    ip_match = re.search(r'\d+\.\d+\.\d+\.\d+', fact)
                    if ip_match:
                        G.nodes[node_id]['ip'] = ip_match.group(0)
            
            # Add arcs (edges)
            for arc in root.findall('./arcs/arc'):
                src_id = int(arc.get('src'))
                dst_id = int(arc.get('dst'))
                G.add_edge(src_id, dst_id)
            
            logger.info(f"Parsed attack graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
            return G
            
        except ET.ParseError as e:
            logger.error(f"Failed to parse XML: {str(e)}")
            raise ValueError(f"Failed to parse attack graph XML: {str(e)}")
        except Exception as e:
            logger.error(f"Error parsing attack graph: {str(e)}")
            raise ValueError(f"Error parsing attack graph: {str(e)}")
    
    def identify_critical_nodes(self, G: nx.DiGraph) -> List[int]:
        """
        Identify critical nodes in the attack graph.
        
        Critical nodes are defined as nodes that, if removed, would
        disconnect the attack paths to the goal nodes.
        
        Args:
            G: NetworkX DiGraph representing the attack graph
            
        Returns:
            List of node IDs that are considered critical
        """
        logger.info("Identifying critical nodes in attack graph")
        
        # Find goal nodes (typically nodes with no outgoing edges or specific types)
        goal_nodes = []
        for node, data in G.nodes(data=True):
            fact = data.get('fact', '')
            if ('execCode' in fact or 'privilege' in fact) and G.out_degree(node) == 0:
                goal_nodes.append(node)
        
        if not goal_nodes:
            logger.warning("No goal nodes found in attack graph")
            # If no clear goal nodes, use nodes with no outgoing edges
            goal_nodes = [node for node, out_degree in G.out_degree() if out_degree == 0]
        
        logger.info(f"Identified {len(goal_nodes)} goal nodes: {goal_nodes}")
        
        # Find all source nodes (nodes with no incoming edges)
        source_nodes = [node for node, in_degree in G.in_degree() if in_degree == 0]
        logger.info(f"Identified {len(source_nodes)} source nodes")
        
        # Find critical nodes using betweenness centrality
        betweenness = nx.betweenness_centrality(G)
        
        # Sort nodes by betweenness centrality
        critical_candidates = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)
        
        # Take top nodes or nodes above a threshold
        critical_nodes = []
        threshold = max(betweenness.values()) * 0.5 if betweenness else 0
        for node, score in critical_candidates:
            if score > threshold:
                critical_nodes.append(node)
        
        # Ensure we have at least some critical nodes
        if not critical_nodes and critical_candidates:
            critical_nodes = [critical_candidates[0][0]]
        
        # Verify with node connectivity between sources and goals
        verified_critical = []
        if source_nodes and goal_nodes:
            G_copy = G.copy()
            for node in critical_nodes:
                G_copy.remove_node(node)
                # Check if removal disconnects any source from any goal
                disconnected = True
                for source in source_nodes:
                    if source in G_copy.nodes():
                        for goal in goal_nodes:
                            if goal in G_copy.nodes():
                                try:
                                    path = nx.shortest_path(G_copy, source, goal)
                                    # If path exists, this node isn't critical for this pair
                                    disconnected = False
                                    break
                                except nx.NetworkXNoPath:
                                    # No path exists
                                    continue
                                
                if disconnected:
                    verified_critical.append(node)
                G_copy = G.copy()
        
        # If verification removed too many nodes, use the original critical nodes
        if len(verified_critical) < len(critical_nodes) * 0.5:
            logger.info(f"Using original critical nodes list of length {len(critical_nodes)}")
            return critical_nodes
        
        logger.info(f"Identified {len(verified_critical)} critical nodes")
        return verified_critical


class GraphEncoder:
    """
    Converts attack graphs and network data into graph embeddings for ML.
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or CONFIG
    
    def encode_attack_graph(self, G: nx.DiGraph) -> Dict:
        """
        Encode attack graph into features suitable for GNN.
        
        Args:
            G: NetworkX DiGraph representing the attack graph
            
        Returns:
            Dictionary with node features, edge index, and edge features
        """
        logger.info(f"Encoding attack graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        # Create node feature matrix
        node_types = {'AND': 0, 'OR': 1, 'LEAF': 2, 'UNKNOWN': 3}
        node_features = []
        node_mapping = {}  # Map node IDs to consecutive indices
        
        for i, (node, data) in enumerate(G.nodes(data=True)):
            node_mapping[node] = i
            
            # Extract node type
            node_type = data.get('type', 'UNKNOWN')
            type_one_hot = [0] * len(node_types)
            type_one_hot[node_types.get(node_type, node_types['UNKNOWN'])] = 1
            
            # Check if node is vulnerability
            is_vulnerability = 1 if 'vulExists' in data.get('fact', '') else 0
            
            # Check if node is host
            is_host = 1 if 'hostAccess' in data.get('fact', '') else 0
            
            # Create feature vector
            features = type_one_hot + [is_vulnerability, is_host]
            
            # Add severity if available
            severity = 0.0
            if 'cve' in data:
                # In a real system, we would look up the CVE severity
                severity = 5.0  # Default medium severity
            
            features.append(severity)
            node_features.append(features)
        
        # Convert to tensor format
        node_features_np = np.array(node_features, dtype=np.float32)
        
        # Create edge index (source, target pairs)
        edge_index = []
        for u, v in G.edges():
            edge_index.append([node_mapping[u], node_mapping[v]])

        # Make sure edge_index is correct for PyTorch Geometric (should be [2, num_edges])
        edge_index_np = np.array(edge_index, dtype=np.int64)
        if edge_index_np.shape[0] > 0:  # Only transpose if not empty
            if edge_index_np.shape[1] == 2:  # If in [num_edges, 2] format
                edge_index_np = edge_index_np.T  # Change to [2, num_edges]
        
        # Add simple edge features (all 1s for now)
        edge_features_np = np.ones((len(edge_index), 1), dtype=np.float32)
        
        # Create mapping back to original node IDs
        reverse_mapping = {v: k for k, v in node_mapping.items()}
        
        return {
            'node_features': node_features_np,
            'edge_index': edge_index_np,
            'edge_features': edge_features_np,
            'node_mapping': node_mapping,
            'reverse_mapping': reverse_mapping,
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges()
        }
    
    def encode_network_data(self, network_flows: pd.DataFrame) -> Dict:
        """
        Encode network flow data into graph format.
        
        Args:
            network_flows: DataFrame with network flow data
            
        Returns:
            Dictionary with node features, edge index, and edge features
        """
        logger.info("Encoding network flow data into graph format")
        
        # Create a graph where nodes are IP addresses and edges are connections
        G = nx.DiGraph()
        
        # Map IP addresses to consistent indices
        all_ips = set(network_flows['src_ip'].unique()) | set(network_flows['dst_ip'].unique())
        ip_to_idx = {ip: i for i, ip in enumerate(all_ips)}
        
        # Add nodes
        for ip in all_ips:
            # Count flows from this IP
            outgoing = network_flows[network_flows['src_ip'] == ip].shape[0]
            incoming = network_flows[network_flows['dst_ip'] == ip].shape[0]
            
            # Basic node features: [outgoing_count, incoming_count, total_count]
            G.add_node(ip_to_idx[ip], 
                      ip=ip, 
                      outgoing=outgoing, 
                      incoming=incoming, 
                      total=outgoing+incoming)
        
        # Add edges with features
        for _, flow in network_flows.iterrows():
            src = ip_to_idx[flow['src_ip']]
            dst = ip_to_idx[flow['dst_ip']]
            
            # Extract edge features
            protocol = flow.get('protocol', 0)
            try:
                protocol = int(protocol) if not pd.isna(protocol) else 0
            except (ValueError, TypeError):
                protocol = 0
                
            src_port = flow.get('src_port', 0)
            try:
                src_port = int(src_port) if not pd.isna(src_port) else 0
            except (ValueError, TypeError):
                src_port = 0
                
            dst_port = flow.get('dst_port', 0)
            try:
                dst_port = int(dst_port) if not pd.isna(dst_port) else 0
            except (ValueError, TypeError):
                dst_port = 0
                
            bytes_sent = flow.get('bytes', 0)
            try:
                bytes_sent = int(bytes_sent) if not pd.isna(bytes_sent) else 0
            except (ValueError, TypeError):
                bytes_sent = 0
            
            # Add edge or update weight if it exists
            if G.has_edge(src, dst):
                G[src][dst]['weight'] += 1
                G[src][dst]['bytes'] += bytes_sent
            else:
                G.add_edge(src, dst, 
                          weight=1, 
                          protocol=protocol,
                          src_port=src_port,
                          dst_port=dst_port,
                          bytes=bytes_sent)
        
        # Convert to format suitable for GNN
        node_features = []
        for i in range(len(all_ips)):
            node = G.nodes[i]
            # Features: [outgoing_count, incoming_count, total_count]
            node_features.append([
                node.get('outgoing', 0),
                node.get('incoming', 0),
                node.get('total', 0)
            ])
        
        node_features_np = np.array(node_features, dtype=np.float32)
        
        # Extract edge information
        edge_index = []
        edge_features = []
        
        for src, dst, data in G.edges(data=True):
            edge_index.append([src, dst])
            edge_features.append([
                data.get('weight', 1),
                data.get('protocol', 0),
                data.get('bytes', 0)
            ])
        
        edge_index_np = np.array(edge_index, dtype=np.int64).T if edge_index else np.zeros((2, 0), dtype=np.int64)
        edge_features_np = np.array(edge_features, dtype=np.float32) if edge_features else np.zeros((0, 3), dtype=np.float32)
        
        return {
            'node_features': node_features_np,
            'edge_index': edge_index_np,
            'edge_features': edge_features_np,
            'ip_to_idx': ip_to_idx,
            'idx_to_ip': {v: k for k, v in ip_to_idx.items()},
            'num_nodes': len(all_ips),
            'num_edges': G.number_of_edges()
        }
    
    def combine_graphs(self, attack_graph_encoding: Dict, network_data_encoding: Dict) -> Dict:
        """
        Combine attack graph and network data graph into a unified graph representation.
        
        Args:
            attack_graph_encoding: Dictionary from encode_attack_graph
            network_data_encoding: Dictionary from encode_network_data
            
        Returns:
            Combined graph encoding
        """
        logger.info("Combining attack graph and network data encodings")
        
        # Number of nodes in each graph
        n_attack = attack_graph_encoding['num_nodes']
        n_network = network_data_encoding['num_nodes']
        
        # Create combined node features
        attack_node_features = attack_graph_encoding['node_features']
        network_node_features = network_data_encoding['node_features']
        
        # Pad features to the same dimension
        attack_feat_dim = attack_node_features.shape[1]
        network_feat_dim = network_node_features.shape[1]
        
        max_feat_dim = max(attack_feat_dim, network_feat_dim)
        
        # Pad attack graph features if needed
        if attack_feat_dim < max_feat_dim:
            padding = np.zeros((n_attack, max_feat_dim - attack_feat_dim), dtype=np.float32)
            attack_node_features = np.hstack([attack_node_features, padding])
        
        # Pad network graph features if needed
        if network_feat_dim < max_feat_dim:
            padding = np.zeros((n_network, max_feat_dim - network_feat_dim), dtype=np.float32)
            network_node_features = np.hstack([network_node_features, padding])
        
        # Concatenate node features
        combined_node_features = np.vstack([attack_node_features, network_node_features])
        
        # Adjust edge indices for the combined graph
        attack_edge_index = attack_graph_encoding['edge_index']
        network_edge_index = network_data_encoding['edge_index']
        
        # Offset network edge indices by the number of attack graph nodes
        if network_edge_index.size > 0:  # Check if there are any edges
            network_edge_index = network_edge_index.copy()
            network_edge_index += n_attack
        
        # Concatenate edge indices
        combined_edge_index = np.hstack([attack_edge_index, network_edge_index]) if network_edge_index.size > 0 else attack_edge_index
        
        # Combine edge features
        attack_edge_features = attack_graph_encoding['edge_features']
        network_edge_features = network_data_encoding['edge_features']
        
        # Adjust dimensions if needed
        attack_edge_feat_dim = attack_edge_features.shape[1]
        network_edge_feat_dim = network_edge_features.shape[1]
        
        max_edge_feat_dim = max(attack_edge_feat_dim, network_edge_feat_dim)
        
        # Pad attack graph edge features if needed
        if attack_edge_feat_dim < max_edge_feat_dim and attack_edge_features.size > 0:
            padding = np.zeros((attack_edge_features.shape[0], max_edge_feat_dim - attack_edge_feat_dim), dtype=np.float32)
            attack_edge_features = np.hstack([attack_edge_features, padding])
        
        # Pad network graph edge features if needed
        if network_edge_feat_dim < max_edge_feat_dim and network_edge_features.size > 0:
            padding = np.zeros((network_edge_features.shape[0], max_edge_feat_dim - network_edge_feat_dim), dtype=np.float32)
            network_edge_features = np.hstack([network_edge_features, padding])
        
        # Concatenate edge features
        combined_edge_features = np.vstack([attack_edge_features, network_edge_features]) if network_edge_features.size > 0 else attack_edge_features
        
        # Create node type marker (0 for attack graph, 1 for network graph)
        node_types = np.zeros(n_attack + n_network, dtype=np.int64)
        node_types[n_attack:] = 1
        
        # Create mappings
        attack_reverse_mapping = attack_graph_encoding.get('reverse_mapping', {})
        network_idx_to_ip = network_data_encoding.get('idx_to_ip', {})
        
        # Combined mapping
        combined_mapping = {}
        for i in range(n_attack):
            if i in attack_reverse_mapping:
                combined_mapping[i] = {'type': 'attack', 'id': attack_reverse_mapping[i]}
        
        for i in range(n_network):
            combined_idx = i + n_attack
            if i in network_idx_to_ip:
                combined_mapping[combined_idx] = {'type': 'network', 'ip': network_idx_to_ip[i]}
        
        return {
            'node_features': combined_node_features,
            'edge_index': combined_edge_index,
            'edge_features': combined_edge_features,
            'node_types': node_types,
            'num_nodes': n_attack + n_network,
            'num_edges': attack_graph_encoding['num_edges'] + network_data_encoding['num_edges'],
            'combined_mapping': combined_mapping,
            'attack_nodes': n_attack,
            'network_nodes': n_network
        }
    
    def create_temporal_graph_sequence(self, graph_encodings: List[Dict], window_size: int = 5) -> List[Dict]:
        """
        Create a sequence of temporal graphs for dynamic GNN.
        
        Args:
            graph_encodings: List of graph encodings ordered by time
            window_size: Number of frames to include in each sequence
            
        Returns:
            List of temporal graph sequences
        """
        logger.info(f"Creating temporal graph sequence with window size {window_size}")
        
        # Ensure we have enough graphs
        if len(graph_encodings) < window_size:
            logger.warning(f"Not enough graph frames ({len(graph_encodings)}) for window size {window_size}")
            # Pad with copies if needed
            if len(graph_encodings) > 0:
                graph_encodings = graph_encodings + [graph_encodings[-1]] * (window_size - len(graph_encodings))
            else:
                logger.error("No graph encodings provided")
                return []
        
        # Create sliding window sequences
        sequences = []
        for i in range(len(graph_encodings) - window_size + 1):
            sequence = graph_encodings[i:i+window_size]
            sequences.append({
                'sequence': sequence,
                'start_idx': i,
                'end_idx': i + window_size - 1
            })
        
        logger.info(f"Created {len(sequences)} temporal graph sequences")
        return sequences


class DataPreprocessor:
    """Preprocess and combine data from different sources for ML training."""
    
    def __init__(self, config: Dict = None):
        self.config = config or CONFIG
        self.data_dir = Path(self.config["data_dir"])
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize feature matrix to have zero mean and unit variance.
        
        Args:
            features: Feature matrix to normalize
            
        Returns:
            Normalized feature matrix
        """
        # Check if feature matrix is not empty
        if features.size == 0:
            return features
        
        # Calculate mean and std along each feature dimension
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        
        # Replace zero std with 1 to avoid division by zero
        std[std == 0] = 1
        
        # Normalize
        normalized_features = (features - mean) / std
        
        return normalized_features
    
    def preprocess_graph_encoding(self, encoding: Dict) -> Dict:
        """
        Preprocess graph encoding to prepare for model input.
        
        Args:
            encoding: Graph encoding from GraphEncoder
            
        Returns:
            Preprocessed graph encoding
        """
        # Normalize node features
        if 'node_features' in encoding and encoding['node_features'].size > 0:
            encoding['node_features'] = self.normalize_features(encoding['node_features'])
        
        # Normalize edge features
        if 'edge_features' in encoding and encoding['edge_features'].size > 0:
            encoding['edge_features'] = self.normalize_features(encoding['edge_features'])
        
        return encoding
    
    def create_training_dataset(self, 
                              graph_encodings: List[Dict], 
                              labels: List[int], 
                              test_split: float = 0.2, 
                              validation_split: float = 0.1) -> Dict:
        """
        Create training, validation, and test datasets.
        
        Args:
            graph_encodings: List of preprocessed graph encodings
            labels: List of labels for each graph encoding
            test_split: Fraction of data to use for testing
            validation_split: Fraction of training data to use for validation
            
        Returns:
            Dictionary with train, validation, and test datasets
        """
        logger.info(f"Creating training dataset with {len(graph_encodings)} samples")
        
        # Ensure we have the same number of encodings and labels
        if len(graph_encodings) != len(labels):
            raise ValueError(f"Number of graph encodings ({len(graph_encodings)}) does not match number of labels ({len(labels)})")
        
        # Shuffle data with fixed random seed for reproducibility
        indices = np.arange(len(graph_encodings))
        np.random.seed(42)
        np.random.shuffle(indices)
        
        # Split into train+validation and test sets
        test_size = int(len(indices) * test_split)
        test_indices = indices[:test_size]
        train_val_indices = indices[test_size:]
        
        # Split train+validation into train and validation sets
        validation_size = int(len(train_val_indices) * validation_split)
        val_indices = train_val_indices[:validation_size]
        train_indices = train_val_indices[validation_size:]
        
        # Create datasets
        train_data = {
            'encodings': [graph_encodings[i] for i in train_indices],
            'labels': [labels[i] for i in train_indices],
            'indices': train_indices.tolist()
        }
        
        val_data = {
            'encodings': [graph_encodings[i] for i in val_indices],
            'labels': [labels[i] for i in val_indices],
            'indices': val_indices.tolist()
        }
        
        test_data = {
            'encodings': [graph_encodings[i] for i in test_indices],
            'labels': [labels[i] for i in test_indices],
            'indices': test_indices.tolist()
        }
        
        logger.info(f"Created datasets with sizes: train={len(train_data['encodings'])}, "
                   f"validation={len(val_data['encodings'])}, test={len(test_data['encodings'])}")
        
        return {
            'train': train_data,
            'validation': val_data,
            'test': test_data,
            'all_indices': indices.tolist()
        }
    
    def save_dataset(self, dataset: Dict, name: str) -> str:
        """
        Save dataset to disk.
        
        Args:
            dataset: Dataset dictionary to save
            name: Name of the dataset
            
        Returns:
            Path to saved dataset
        """
        dataset_dir = self.data_dir / "datasets"
        dataset_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = dataset_dir / f"{name}_{timestamp}.pkl"
        
        with open(save_path, 'wb') as f:
            pickle.dump(dataset, f)
        
        logger.info(f"Saved dataset {name} to {save_path}")
        return str(save_path)
    
    def load_dataset(self, path: str) -> Dict:
        """
        Load saved dataset from disk.
        
        Args:
            path: Path to saved dataset
            
        Returns:
            Loaded dataset dictionary
        """
        try:
            with open(path, 'rb') as f:
                dataset = pickle.load(f)
            
            logger.info(f"Loaded dataset from {path}")
            return dataset
        except Exception as e:
            logger.error(f"Failed to load dataset from {path}: {str(e)}")
            raise ValueError(f"Failed to load dataset from {path}: {str(e)}")


#######################
# ML ENGINE LAYER
#######################

if HAS_TORCH:
    
    class GraphDataset(Dataset):
        """PyTorch Dataset for graph data."""
        
        def __init__(self, encodings: List[Dict], labels: List[int]):
            """
            Initialize graph dataset.
            
            Args:
                encodings: List of graph encodings
                labels: List of labels for each encoding
            """
            self.encodings = encodings
            self.labels = labels
            
            # Validate inputs
            if len(encodings) != len(labels):
                raise ValueError(f"Number of encodings ({len(encodings)}) does not match number of labels ({len(labels)})")
        
        def __len__(self) -> int:
            return len(self.encodings)
        
        def __getitem__(self, idx: int) -> Tuple[Dict, int]:
            encoding = self.encodings[idx]
            
            # Convert numpy arrays to torch tensors
            result = {}
            for key, value in encoding.items():
                if isinstance(value, np.ndarray):
                    if key == 'edge_index':
                        # Ensure edge_index is in the correct format [2, num_edges]
                        if value.shape[0] > 0:
                            if value.shape[0] == 2:
                                # Already in correct format
                                result[key] = torch.tensor(value, dtype=torch.long)
                            else:
                                # Need to transpose
                                result[key] = torch.tensor(value.T, dtype=torch.long)
                        else:
                            # Empty edge index
                            result[key] = torch.zeros((2, 0), dtype=torch.long)
                    elif key == 'node_features' or key == 'edge_features':
                        result[key] = torch.tensor(value, dtype=torch.float)
                    else:
                        result[key] = torch.tensor(value)
                else:
                    result[key] = value
                    
            return result, self.labels[idx]



    class TemporalGraphDataset(Dataset):
        """PyTorch Dataset for temporal graph data."""
        
        def __init__(self, sequences: List[Dict], labels: List[int]):
            """
            Initialize temporal graph dataset.
            
            Args:
                sequences: List of graph sequence dictionaries
                labels: List of labels for each sequence
            """
            self.sequences = sequences
            self.labels = labels
            
            # Validate inputs
            if len(sequences) != len(labels):
                raise ValueError(f"Number of sequences ({len(sequences)}) does not match number of labels ({len(labels)})")
        
        def __len__(self) -> int:
            return len(self.sequences)
        
        def __getitem__(self, idx: int) -> Tuple[Dict, int]:
            return self.sequences[idx], self.labels[idx]


    class DynamicGNN(nn.Module):
        """Dynamic Graph Neural Network for threat detection."""
        
        def __init__(self, input_dim: int, hidden_channels: int, num_classes: int, num_layers: int = 3, dropout: float = 0.2):
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
        
        def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor = None) -> torch.Tensor:
            """
            Forward pass for a single graph.
            
            Args:
                x: Node feature matrix
                edge_index: Graph connectivity in COO format
                batch: Batch vector for multiple graphs (None for single graph)
                
            Returns:
                Output predictions
            """
            # Log input sizes for debugging
            logger.debug(f"Input x shape: {x.shape}, edge_index shape: {edge_index.shape}")
            if batch is not None:
                logger.debug(f"Batch shape: {batch.shape}")
                
            # Initial feature transformation
            x = self.feature_transform(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # GNN layers
            for i in range(self.num_layers):
                # Ensure edge_index has the right format
                if edge_index.dim() > 2:
                    logger.error(f"Invalid edge_index dimension: {edge_index.dim()}")
                    # Try to reshape it
                    edge_index = edge_index.view(2, -1)
                    
                x = self.convs[i](x, edge_index)
                
                # Apply batch norm - handle different batch sizes
                if x.size(0) > 1:  # More than one node
                    x = self.batch_norms[i](x)
                    
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Graph-level readout (if batch is provided)
            if batch is not None:
                x = global_mean_pool(x, batch)
            else:
                x = torch.mean(x, dim=0, keepdim=True)
            
            # Classification
            x = self.classifier(x)
            
            return x
        
        def forward_temporal(self, 
                           x_sequence: List[torch.Tensor], 
                           edge_index_sequence: List[torch.Tensor], 
                           batch_sequence: List[torch.Tensor] = None) -> torch.Tensor:
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
                
                # Graph-level readout
                if batch is not None:
                    pooled_x = global_mean_pool(x, batch)
                else:
                    pooled_x = torch.mean(x, dim=0, keepdim=True)
                
                embeddings.append(pooled_x)
            
            # Stack embeddings from all time steps
            embeddings = torch.stack(embeddings, dim=1)  # Shape: [batch_size, sequence_length, hidden_channels]
            
            # Apply temporal attention
            attention_scores = self.temporal_attention(embeddings)  # Shape: [batch_size, sequence_length, 1]
            attention_weights = F.softmax(attention_scores, dim=1)  # Shape: [batch_size, sequence_length, 1]
            
            # Weighted sum of embeddings
            context = torch.sum(attention_weights * embeddings, dim=1)  # Shape: [batch_size, hidden_channels]
            
            # Classification
            output = self.classifier(context)
            
            return output


    class EvolveGCN(nn.Module):
        """
        Implementation of EvolveGCN model for temporal graph learning.
        
        Based on "EvolveGCN: Evolving Graph Convolutional Networks for Dynamic Graphs"
        """
        
        def __init__(self, input_dim: int, hidden_channels: int, num_classes: int, num_layers: int = 2, dropout: float = 0.2):
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
                # Initialized according to Glorot initialization
                stdv = 1. / math.sqrt(in_channels)
                weight = nn.Parameter(torch.FloatTensor(in_channels, out_channels).uniform_(-stdv, stdv))
                self.weights.append(weight)
                
                # GRU cell for evolving the weights
                self.gru_cells.append(nn.GRUCell(out_channels, in_channels))
            
            # Batch normalization layers
            self.batch_norms = nn.ModuleList()
            for _ in range(num_layers):
                self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
            
            # Output classification layer
            self.classifier = nn.Linear(hidden_channels, num_classes)
        
        def forward(self, x_sequence: List[torch.Tensor], edge_index_sequence: List[torch.Tensor], 
                   batch_sequence: List[torch.Tensor] = None) -> torch.Tensor:
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
            for t in range(sequence_length):
                x_sequence[t] = self.feature_transform(x_sequence[t])
            
            # Process the sequence
            for l in range(self.num_layers):
                # Get initial weight for this layer
                weight = self.weights[l]
                
                # Create hidden state for GRU (for each layer)
                h = weight.view(1, -1)  # Shape: [1, in_channels * out_channels]
                
                # Process the sequence for this layer
                for t in range(sequence_length):
                    # Get current graph
                    x = x_sequence[t]
                    edge_index = edge_index_sequence[t]
                    
                    # Use current weight matrix for GCN operation
                    weight_reshaped = h.view(weight.shape)
                    
                    # Compute message passing using GCN
                    # Simplified version of GCN message passing
                    row, col = edge_index
                    deg = degree(row, x.size(0), dtype=x.dtype)
                    deg_inv_sqrt = deg.pow(-0.5)
                    norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
                    
                    # GCN convolution (mat_mul and aggregation combined)
                    support = torch.mm(x, weight_reshaped)
                    output = torch.zeros_like(support)
                    for i in range(edge_index.size(1)):
                        src, dst = col[i], row[i]
                        output[dst] += support[src] * norm[i]
                    
                    # Apply batch normalization
                    output = self.batch_norms[l](output)
                    
                    # Nonlinearity
                    output = F.relu(output)
                    
                    # Apply dropout
                    output = F.dropout(output, p=self.dropout, training=self.training)
                    
                    # Update graph for next layer/timestep
                    x_sequence[t] = output
                    
                    # Update the weight using GRU
                    # We take the mean of the node embeddings as the "input" to the GRU
                    if t < sequence_length - 1:
                        input_to_gru = torch.mean(output, dim=0, keepdim=True)  # Shape: [1, hidden_channels]
                        h = self.gru_cells[l](input_to_gru, h)  # Shape: [1, in_channels * out_channels]
            
            # Final graph readout from the last graph in the sequence
            final_x = x_sequence[-1]
            final_batch = batch_sequence[-1] if batch_sequence is not None else None
            
            if final_batch is not None:
                pooled = global_mean_pool(final_x, final_batch)
            else:
                pooled = torch.mean(final_x, dim=0, keepdim=True)
            
            # Classification
            output = self.classifier(pooled)
            
            return output


    class DySAT(nn.Module):
        """
        Implementation of DySAT model for dynamic graph representation learning.
        
        Based on "DySAT: Deep Neural Representation Learning on Dynamic Graphs via
        Self-Attention Networks"
        """
        
        def __init__(self, input_dim: int, hidden_channels: int, num_classes: int, 
                    num_heads: int = 4, num_layers: int = 2, dropout: float = 0.2):
            """
            Initialize DySAT model.
            
            Args:
                input_dim: Input feature dimension
                hidden_channels: Number of hidden units
                num_classes: Number of output classes
                num_heads: Number of attention heads
                num_layers: Number of layers
                dropout: Dropout probability
            """
            super(DySAT, self).__init__()
            
            self.input_dim = input_dim
            self.hidden_channels = hidden_channels
            self.num_classes = num_classes
            self.num_heads = num_heads
            self.num_layers = num_layers
            self.dropout = dropout
            
            # Feature transformation
            self.feature_transform = nn.Linear(input_dim, hidden_channels)
            
            # Structural attention layers (for each timestep)
            self.structural_attn = nn.ModuleList()
            for _ in range(num_layers):
                self.structural_attn.append(
                    GATConv(hidden_channels, hidden_channels // num_heads, heads=num_heads, dropout=dropout)
                )
            
            # Temporal attention
            self.temporal_attn = nn.ModuleList()
            for _ in range(num_layers):
                # Multi-head attention for temporal dimension
                self.temporal_attn.append(nn.MultiheadAttention(
                    embed_dim=hidden_channels,
                    num_heads=num_heads,
                    dropout=dropout
                ))
            
            # Output classification layer
            self.classifier = nn.Linear(hidden_channels, num_classes)
            
            # Layer normalization
            self.layer_norms_struct = nn.ModuleList()
            self.layer_norms_temp = nn.ModuleList()
            for _ in range(num_layers):
                self.layer_norms_struct.append(nn.LayerNorm(hidden_channels))
                self.layer_norms_temp.append(nn.LayerNorm(hidden_channels))
        
        def forward(self, x_sequence: List[torch.Tensor], edge_index_sequence: List[torch.Tensor], 
                   batch_sequence: List[torch.Tensor] = None) -> torch.Tensor:
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
            
            # Process each graph first with structural attention
            struct_embeddings = []
            
            for t in range(sequence_length):
                x = x_sequence[t]
                edge_index = edge_index_sequence[t]
                
                # Initial transformation
                h = self.feature_transform(x)
                
                # Structural attention layers
                for i in range(self.num_layers):
                    # Apply GAT convolution
                    h_strutural = self.structural_attn[i](h, edge_index)
                    
                    # Apply layer normalization
                    h_strutural = self.layer_norms_struct[i](h_strutural)
                    
                    # Add residual connection and apply non-linearity
                    h = h + F.dropout(h_strutural, p=self.dropout, training=self.training)
                    h = F.relu(h)
                
                # Store the embeddings for this timestep
                struct_embeddings.append(h)
            
            # Process with temporal attention
            # First, stack the embeddings from all timesteps
            # For each node, we need to process its temporal sequence
            
            # Determine the maximum number of nodes
            max_nodes = max(emb.size(0) for emb in struct_embeddings)
            
            # Pad embeddings to the same size
            padded_embeddings = []
            mask = []
            
            for emb in struct_embeddings:
                num_nodes = emb.size(0)
                padding = torch.zeros(max_nodes - num_nodes, emb.size(1), device=emb.device)
                padded_emb = torch.cat([emb, padding], dim=0)
                padded_embeddings.append(padded_emb)
                
                # Create attention mask
                m = torch.ones(max_nodes, device=emb.device)
                m[num_nodes:] = 0
                mask.append(m)
            
            # Stack to create a tensor of shape [sequence_length, max_nodes, hidden_channels]
            stacked_embeddings = torch.stack(padded_embeddings, dim=0)
            mask = torch.stack(mask, dim=0).bool()
            
            # Apply temporal attention for each node
            # First, transpose to shape [max_nodes, sequence_length, hidden_channels]
            stacked_embeddings = stacked_embeddings.transpose(0, 1)
            mask = mask.transpose(0, 1)
            
            temporal_embeddings = []
            
            for i in range(max_nodes):
                node_temporal_seq = stacked_embeddings[i].unsqueeze(1)  # [sequence_length, 1, hidden_channels]
                node_mask = mask[i]
                
                # Skip nodes that were padded
                if not node_mask.any():
                    continue
                
                # Process through temporal attention layers
                h = node_temporal_seq
                
                for j in range(self.num_layers):
                    # Apply temporal attention
                    attn_output, _ = self.temporal_attn[j](
                        h, h, h,
                        key_padding_mask=~node_mask.unsqueeze(0)  # PyTorch uses key_padding_mask where 1 means ignore
                    )
                    
                    # Apply layer normalization
                    attn_output = self.layer_norms_temp[j](attn_output)
                    
                    # Add residual connection and apply non-linearity
                    h = h + F.dropout(attn_output, p=self.dropout, training=self.training)
                    h = F.relu(h)
                
                # Get the embedding for the last timestep
                temporal_embeddings.append(h[-1])
            
            temporal_embeddings = torch.cat(temporal_embeddings, dim=0)
            
            # Handle batched graphs
            if batch_sequence is not None:
                # Use the batch vector from the last timestep
                final_batch = batch_sequence[-1]
                # Only keep the nodes that were in the original graph
                final_batch = final_batch[:temporal_embeddings.size(0)]
                pooled = global_mean_pool(temporal_embeddings, final_batch)
            else:
                pooled = torch.mean(temporal_embeddings, dim=0, keepdim=True)
            
            # Classification
            output = self.classifier(pooled)
            
            return output


    class ModelTrainer:
        """Trainer for graph neural network models."""
        
        def __init__(self, model: nn.Module, config: Dict = None):
            """
            Initialize the model trainer.
            
            Args:
                model: PyTorch model to train
                config: Configuration dictionary
            """
            self.model = model
            self.config = config or CONFIG
            self.device = torch.device('cuda' if self.config['use_gpu'] and torch.cuda.is_available() else 'cpu')
            
            # Move model to device
            self.model = self.model.to(self.device)
            
            logger.info(f"Using device: {self.device}")
            
            # Initialize optimizer
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config['model_config']['learning_rate'],
                weight_decay=self.config['model_config']['weight_decay']
            )
            
            # Learning rate scheduler
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
            )
            
            # Track best model
            self.best_val_loss = float('inf')
            self.best_model_state = None
            self.no_improve_count = 0
            
            # Set models directory
            self.models_dir = Path(self.config['models_dir'])
            self.models_dir.mkdir(parents=True, exist_ok=True)
        
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
            
            for batch_idx, (data, targets) in enumerate(train_loader):
                # Extract graph data
                if isinstance(data, dict) and 'sequence' in data:
                    # Handle temporal graph sequences
                    x_sequence = [x['node_features'].float().to(self.device) for x in data['sequence']]
                    edge_index_sequence = [x['edge_index'].long().to(self.device) for x in data['sequence']]
                    batch_sequence = None
                    
                    # Forward pass
                    outputs = self.model.forward_temporal(x_sequence, edge_index_sequence, batch_sequence)
                else:
                    # Handle single graphs
                    x = data['node_features'].float().to(self.device)
                    edge_index = data['edge_index'].long().to(self.device)
                    
                    # Debug info
                    logger.debug(f"x shape: {x.shape}, edge_index shape: {edge_index.shape}")
                    
                    # For batched processing, we need to properly format the data
                    # In a batch, edge_index should be a tensor of shape [2, total_edges]
                    if len(edge_index.shape) == 3:  # Batched edge indices
                        # This means we have a batch dimension
                        batch_size = edge_index.shape[0]
                        
                        # We need to create a global edge index and a batch vector
                        global_edge_index = []
                        batch_vector = []
                        node_offset = 0
                        
                        for b in range(batch_size):
                            # Get edges for this batch item
                            batch_edges = edge_index[b]
                            if batch_edges.shape[1] > 0:  # If there are edges
                                # Add offset to the node indices
                                offset_edges = batch_edges.clone()
                                offset_edges = offset_edges + node_offset
                                
                                # Add to global edge list
                                global_edge_index.append(offset_edges)
                                
                            # Create batch vector for this batch item's nodes
                            num_nodes = x[b].shape[0]
                            batch_vector.extend([b] * num_nodes)
                            
                            # Update node offset for next batch item
                            node_offset += num_nodes
                        
                        if global_edge_index:
                            # Concatenate all edge indices
                            edge_index = torch.cat(global_edge_index, dim=1)
                        else:
                            # No edges in the batch
                            edge_index = torch.zeros((2, 0), dtype=torch.long, device=self.device)
                        
                        # Create batch vector tensor
                        batch = torch.tensor(batch_vector, dtype=torch.long, device=self.device)
                        
                        # Flatten node features
                        x = x.view(-1, x.shape[-1])
                        
                    else:
                        batch = None
                    
                    # Forward pass
                    outputs = self.model(x, edge_index, batch)
                
                # Move targets to device
                targets = torch.tensor(targets).long().to(self.device)
                
                # Compute loss
                loss = F.cross_entropy(outputs, targets)
                
                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Update statistics
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Print progress
                if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
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
                for batch_idx, (data, targets) in enumerate(val_loader):
                    # Extract graph data
                    if isinstance(data, dict) and 'sequence' in data:
                        # Handle temporal graph sequences
                        x_sequence = [x['node_features'].float().to(self.device) for x in data['sequence']]
                        edge_index_sequence = [x['edge_index'].long().to(self.device) for x in data['sequence']]
                        batch_sequence = None
                        
                        # Forward pass
                        outputs = self.model.forward_temporal(x_sequence, edge_index_sequence, batch_sequence)
                    else:
                        # Handle single graphs
                        x = data['node_features'].float().to(self.device)
                        edge_index = data['edge_index'].long().to(self.device)
                        
                        # For batched processing, same as in train_epoch
                        if len(edge_index.shape) == 3:  # Batched edge indices
                            batch_size = edge_index.shape[0]
                            
                            global_edge_index = []
                            batch_vector = []
                            node_offset = 0
                            
                            for b in range(batch_size):
                                batch_edges = edge_index[b]
                                if batch_edges.shape[1] > 0:
                                    offset_edges = batch_edges.clone()
                                    offset_edges = offset_edges + node_offset
                                    global_edge_index.append(offset_edges)
                                
                                num_nodes = x[b].shape[0]
                                batch_vector.extend([b] * num_nodes)
                                node_offset += num_nodes
                            
                            if global_edge_index:
                                edge_index = torch.cat(global_edge_index, dim=1)
                            else:
                                edge_index = torch.zeros((2, 0), dtype=torch.long, device=self.device)
                            
                            batch = torch.tensor(batch_vector, dtype=torch.long, device=self.device)
                            x = x.view(-1, x.shape[-1])
                        else:
                            batch = None
                        
                        # Forward pass
                        outputs = self.model(x, edge_index, batch)
                    
                    # Move targets to device
                    targets = torch.tensor(targets).long().to(self.device)
                    
                    # Compute loss
                    loss = F.cross_entropy(outputs, targets)
                    
                    # Update statistics
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
            
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
                epochs = self.config['model_config']['epochs']
            
            patience = self.config['model_config']['patience']
            
            logger.info(f"Starting training for {epochs} epochs...")
            
            history = {
                'train_loss': [],
                'train_acc': [],
                'val_loss': [],
                'val_acc': []
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
            # Create checkpoint directory
            checkpoint_dir = self.models_dir / "checkpoints"
            checkpoint_dir.mkdir(exist_ok=True)
            
            # Create timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create checkpoint filename
            prefix = "best_" if is_best else ""
            architecture = self.config['model_config']['architecture']
            checkpoint_file = checkpoint_dir / f"{prefix}{architecture}_{timestamp}.pt"
            
            # Create checkpoint dictionary
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_val_loss': self.best_val_loss,
                'config': self.config,
                'architecture': architecture,
                'timestamp': timestamp
            }
            
            # Save checkpoint
            torch.save(checkpoint, checkpoint_file)
            logger.info(f"Saved {'best ' if is_best else ''}checkpoint to {checkpoint_file}")
            
            return str(checkpoint_file)
        
        def load_checkpoint(self, checkpoint_path: str) -> None:
            """
            Load model checkpoint.
            
            Args:
                checkpoint_path: Path to checkpoint file
            """
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                
                # Load model state
                self.model.load_state_dict(checkpoint['model_state_dict'])
                
                # Load optimizer state
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
                # Load other parameters
                self.best_val_loss = checkpoint['best_val_loss']
                
                logger.info(f"Loaded checkpoint from {checkpoint_path}")
                logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
                
            except Exception as e:
                logger.error(f"Failed to load checkpoint from {checkpoint_path}: {str(e)}")
                raise ValueError(f"Failed to load checkpoint: {str(e)}")


    class ModelInference:
        """Class for making predictions with trained models."""
        
        def __init__(self, model: nn.Module, config: Dict = None):
            """
            Initialize model inference.
            
            Args:
                model: Trained PyTorch model
                config: Configuration dictionary
            """
            self.model = model
            self.config = config or CONFIG
            self.device = torch.device('cuda' if self.config['use_gpu'] and torch.cuda.is_available() else 'cpu')
            
            # Move model to device
            self.model = self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Inference using device: {self.device}")
        
        def predict(self, data: Dict) -> Dict:
            """
            Make prediction for a single graph.
            
            Args:
                data: Graph encoding dictionary
                
            Returns:
                Dictionary with prediction results
            """
            self.model.eval()
            
            with torch.no_grad():
                # Extract graph data
                if 'sequence' in data:
                    # Handle temporal graph sequence
                    x_sequence = [torch.tensor(x['node_features']).float().to(self.device) for x in data['sequence']]
                    edge_index_sequence = [torch.tensor(x['edge_index']).long().to(self.device) for x in data['sequence']]
                    batch_sequence = None
                    
                    # Forward pass
                    outputs = self.model.forward_temporal(x_sequence, edge_index_sequence, batch_sequence)
                else:
                    # Handle single graph
                    x = torch.tensor(data['node_features']).float().to(self.device)
                    edge_index = torch.tensor(data['edge_index']).long().to(self.device)
                    batch = None
                    
                    # Forward pass
                    outputs = self.model(x, edge_index, batch)
                
                # Get predicted class and probabilities
                probabilities = F.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                
                # Convert to numpy for JSON serialization
                probabilities = probabilities.cpu().numpy().tolist()[0]
            
            return {
                'predicted_class': predicted_class,
                'probabilities': probabilities,
                'timestamp': datetime.datetime.now().isoformat()
            }
        
        def predict_batch(self, data_loader: DataLoader) -> List[Dict]:
            """
            Make predictions for a batch of graphs.
            
            Args:
                data_loader: DataLoader with graph data
                
            Returns:
                List of prediction dictionaries
            """
            self.model.eval()
            predictions = []
            
            with torch.no_grad():
                for batch_idx, (data, _) in enumerate(data_loader):
                    # Extract graph data
                    if isinstance(data, dict) and 'sequence' in data:
                        # Handle temporal graph sequences
                        x_sequence = [x['node_features'].float().to(self.device) for x in data['sequence']]
                        edge_index_sequence = [x['edge_index'].long().to(self.device) for x in data['sequence']]
                        batch_sequence = None
                        
                        # Forward pass
                        outputs = self.model.forward_temporal(x_sequence, edge_index_sequence, batch_sequence)
                    else:
                        # Handle single graphs
                        x = data['node_features'].float().to(self.device)
                        edge_index = data['edge_index'].long().to(self.device)
                        batch = None
                        
                        # Forward pass
                        outputs = self.model(x, edge_index, batch)
                    
                    # Get predicted classes and probabilities
                    probabilities = F.softmax(outputs, dim=1)
                    predicted_classes = torch.argmax(probabilities, dim=1).cpu().numpy().tolist()
                    probabilities = probabilities.cpu().numpy().tolist()
                    
                    # Create prediction dictionaries
                    for i in range(len(predicted_classes)):
                        predictions.append({
                            'predicted_class': predicted_classes[i],
                            'probabilities': probabilities[i],
                            'timestamp': datetime.datetime.now().isoformat()
                        })
            
            return predictions


    class ExplainabilityModule:
        """Module for explaining GNN model predictions."""
        
        def __init__(self, model: nn.Module, config: Dict = None):
            """
            Initialize explainability module.
            
            Args:
                model: Trained GNN model
                config: Configuration dictionary
            """
            self.model = model
            self.config = config or CONFIG
            self.device = torch.device('cuda' if self.config['use_gpu'] and torch.cuda.is_available() else 'cpu')
            
            # Move model to device
            self.model = self.model.to(self.device)
            self.model.eval()
        
        def explain_prediction(self, data: Dict) -> Dict:
            """
            Explain prediction for a single graph.
            
            Args:
                data: Graph encoding dictionary
                
            Returns:
                Dictionary with explanation
            """
            # Simple gradient-based feature importance
            # In a real implementation, more sophisticated methods would be used
            
            # Extract graph data
            if 'sequence' in data:
                # For temporal graphs, focus on the last graph in the sequence
                last_graph = data['sequence'][-1]
                x = torch.tensor(last_graph['node_features']).float().to(self.device)
                edge_index = torch.tensor(last_graph['edge_index']).long().to(self.device)
            else:
                x = torch.tensor(data['node_features']).float().to(self.device)
                edge_index = torch.tensor(data['edge_index']).long().to(self.device)
            
            # Enable gradient tracking
            x.requires_grad_(True)
            
            # Forward pass
            self.model.eval()
            if 'sequence' in data:
                # For temporal model
                # Create a sequence with a single graph 
                x_sequence = [x]
                edge_index_sequence = [edge_index]
                batch_sequence = None
                outputs = self.model.forward_temporal(x_sequence, edge_index_sequence, batch_sequence)
            else:
                batch = None
                outputs = self.model(x, edge_index, batch)
            
            # Get predicted class
            predicted_class = torch.argmax(outputs, dim=1).item()
            
            # Compute gradients with respect to inputs
            outputs[0, predicted_class].backward()
            
            # Feature importance is the gradient of the output with respect to the input
            feature_importance = x.grad.abs().sum(dim=0).cpu().numpy().tolist()
            
            # Compute node importance by summing gradient magnitudes for each node
            node_importance = x.grad.abs().sum(dim=1).cpu().numpy().tolist()
            
            # Get top important nodes
            top_k = min(5, len(node_importance))
            top_nodes_idx = np.argsort(node_importance)[-top_k:].tolist()
            top_nodes_importance = [node_importance[i] for i in top_nodes_idx]
            
            # Map top nodes back to original IDs if mapping exists
            top_nodes_original = []
            if 'reverse_mapping' in data:
                for idx in top_nodes_idx:
                    if idx in data['reverse_mapping']:
                        top_nodes_original.append(data['reverse_mapping'][idx])
                    else:
                        top_nodes_original.append(idx)
            else:
                top_nodes_original = top_nodes_idx
            
            return {
                'predicted_class': predicted_class,
                'feature_importance': feature_importance,
                'node_importance': node_importance,
                'top_important_nodes': list(zip(top_nodes_original, top_nodes_importance)),
                'explanation_method': 'gradient',
                'timestamp': datetime.datetime.now().isoformat()
            }
        
        def identify_attack_paths(self, data: Dict, G: nx.DiGraph) -> Dict:
            """
            Identify most likely attack paths based on node importance.
            
            Args:
                data: Graph encoding dictionary
                G: Original NetworkX DiGraph
                
            Returns:
                Dictionary with attack paths
            """
            # Get node importance
            explanation = self.explain_prediction(data)
            
            # Create dictionary mapping indices to importance scores
            node_importance = {}
            for i, importance in enumerate(explanation['node_importance']):
                if 'reverse_mapping' in data and i in data['reverse_mapping']:
                    original_id = data['reverse_mapping'][i]
                    node_importance[original_id] = importance
                else:
                    node_importance[i] = importance
            
            # Find goal nodes (nodes with no outgoing edges)
            goal_nodes = []
            for node in G.nodes():
                if G.out_degree(node) == 0:
                    goal_nodes.append(node)
            
            # If no clear goal nodes, use the most important nodes
            if not goal_nodes:
                goal_nodes = [pair[0] for pair in explanation['top_important_nodes']]
            
            # Find source nodes (nodes with no incoming edges)
            source_nodes = []
            for node in G.nodes():
                if G.in_degree(node) == 0:
                    source_nodes.append(node)
            
            # Create a weighted graph where edge weights are inversely proportional to node importance
            G_weighted = G.copy()
            for node in G_weighted.nodes():
                # Set default importance if not available
                importance = node_importance.get(node, 0.1)
                
                # Avoid division by zero
                if importance <= 0:
                    importance = 0.1
                
                # Set node weight inversely proportional to importance
                G_weighted.nodes[node]['weight'] = 1.0 / importance
            
            # Find shortest paths from all sources to all goals
            attack_paths = []
            for source in source_nodes:
                for goal in goal_nodes:
                    try:
                        # Find shortest path based on node weights
                        path = nx.shortest_path(G_weighted, source=source, target=goal, weight='weight')
                        
                        # Calculate path importance as sum of node importances
                        path_importance = sum(node_importance.get(node, 0) for node in path)
                        
                        attack_paths.append({
                            'path': path,
                            'importance': path_importance,
                            'source': source,
                            'goal': goal,
                            'length': len(path)
                        })
                    except nx.NetworkXNoPath:
                        continue
            
            # Sort attack paths by importance
            attack_paths.sort(key=lambda x: x['importance'], reverse=True)
            
            # Return top 5 paths
            top_paths = attack_paths[:5]
            
            return {
                'attack_paths': top_paths,
                'goal_nodes': goal_nodes,
                'source_nodes': source_nodes,
                'timestamp': datetime.datetime.now().isoformat()
            }
        
        def visualize_explanation(self, data: Dict, G: nx.DiGraph, save_path: str = None) -> Dict:
            """
            Create visualization of explanation.
            
            Args:
                data: Graph encoding dictionary
                G: Original NetworkX DiGraph
                save_path: Path to save visualization (if None, will be generated)
                
            Returns:
                Dictionary with paths to visualizations
            """
            # Get node importance
            explanation = self.explain_prediction(data)
            
            # Map node importances back to the original graph
            node_importance = {}
            for i, importance in enumerate(explanation['node_importance']):
                if 'reverse_mapping' in data and i in data['reverse_mapping']:
                    original_id = data['reverse_mapping'][i]
                    node_importance[original_id] = importance
                else:
                    node_importance[i] = importance
            
            # Identify attack paths
            attack_paths_info = self.identify_attack_paths(data, G)
            attack_paths = attack_paths_info['attack_paths']
            
            # Create output directory if not provided
            if save_path is None:
                vis_dir = Path(self.config['data_dir']) / "visualizations"
                vis_dir.mkdir(exist_ok=True)
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = vis_dir / f"explanation_{timestamp}"
            else:
                save_path = Path(save_path)
            
            # Create visualization directory
            save_path.mkdir(exist_ok=True, parents=True)
            
            # Create node importance visualization
            node_importance_path = save_path / "node_importance.png"
            
            # Normalize node importance for visualization
            max_importance = max(node_importance.values()) if node_importance else 1.0
            normalized_importance = {k: v/max_importance for k, v in node_importance.items()}
            
            # Set node colors based on importance
            node_colors = []
            for node in G.nodes():
                importance = normalized_importance.get(node, 0)
                # Use a color gradient from blue (low importance) to red (high importance)
                color = (importance, 0, 1-importance)
                node_colors.append(color)
            
            # Set node sizes based on importance
            node_sizes = []
            for node in G.nodes():
                importance = normalized_importance.get(node, 0)
                size = 100 + 500 * importance
                node_sizes.append(size)
            
            # Create node importance visualization
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(G, seed=42)  # Consistent layout
            nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors)
            nx.draw_networkx_edges(G, pos, alpha=0.6, arrows=True)
            nx.draw_networkx_labels(G, pos, font_size=8)
            plt.title("Node Importance Visualization")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(node_importance_path)
            plt.close()
            
            # Create attack path visualizations
            attack_path_files = []
            for i, path_info in enumerate(attack_paths[:3]):  # Visualize top 3 paths
                path = path_info['path']
                path_file = save_path / f"attack_path_{i+1}.png"
                
                plt.figure(figsize=(12, 8))
                
                # Draw all nodes and edges with low alpha
                nx.draw_networkx_nodes(G, pos, node_size=50, node_color='lightgray', alpha=0.3)
                nx.draw_networkx_edges(G, pos, alpha=0.1, arrows=True)
                
                # Create path graph
                path_edges = list(zip(path[:-1], path[1:]))
                
                # Highlight path nodes
                path_node_colors = []
                for node in path:
                    importance = normalized_importance.get(node, 0)
                    color = (importance, 0, 1-importance)
                    path_node_colors.append(color)
                
                # Draw path with high alpha
                nx.draw_networkx_nodes(G, pos, nodelist=path, node_size=300, node_color=path_node_colors)
                nx.draw_networkx_edges(G, pos, edgelist=path_edges, width=3, alpha=1.0, arrows=True, edge_color='red')
                nx.draw_networkx_labels(G, pos, font_size=8)
                
                plt.title(f"Attack Path {i+1}: Source={path_info['source']}, Goal={path_info['goal']}")
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(path_file)
                plt.close()
                
                attack_path_files.append(str(path_file))
            
            # Return paths to visualizations
            return {
                'node_importance_visualization': str(node_importance_path),
                'attack_path_visualizations': attack_path_files,
                'timestamp': datetime.datetime.now().isoformat()
            }
else:
    # Dummy classes for when PyTorch is not available
    class GraphDataset:
        def __init__(self, *args, **kwargs):
            logger.error("PyTorch not available. Cannot create GraphDataset.")
    
    class TemporalGraphDataset:
        def __init__(self, *args, **kwargs):
            logger.error("PyTorch not available. Cannot create TemporalGraphDataset.")
    
    class DynamicGNN:
        def __init__(self, *args, **kwargs):
            logger.error("PyTorch not available. Cannot create DynamicGNN.")
    
    class EvolveGCN:
        def __init__(self, *args, **kwargs):
            logger.error("PyTorch not available. Cannot create EvolveGCN.")
    
    class DySAT:
        def __init__(self, *args, **kwargs):
            logger.error("PyTorch not available. Cannot create DySAT.")
    
    class ModelTrainer:
        def __init__(self, *args, **kwargs):
            logger.error("PyTorch not available. Cannot create ModelTrainer.")
    
    class ModelInference:
        def __init__(self, *args, **kwargs):
            logger.error("PyTorch not available. Cannot create ModelInference.")
    
    class ExplainabilityModule:
        def __init__(self, *args, **kwargs):
            logger.error("PyTorch not available. Cannot create ExplainabilityModule.")

#######################
# API LAYER
#######################

if HAS_FASTAPI:
    # API Models
    class VulnerabilityInput(BaseModel):
        ip: str
        name: str = Field(..., description="Vulnerability name or description")
        cve: str = Field(None, description="CVE identifier if available")
        severity: float = Field(5.0, description="Severity score (0-10)")

    class HostInput(BaseModel):
        ip: str
        service: str = Field("webServer", description="Service running on the host")
        service_port: str = Field("httpProtocol", description="Port the service is running on")
        user: str = Field("serviceUser", description="User the service runs as")
        role: str = Field("serviceRole", description="Role of the service")
        connections: List[Dict] = Field([], description="Connections to other hosts")

    class NetworkConfigInput(BaseModel):
        hosts: List[HostInput] = Field(..., description="List of hosts in the network")

    class DetectionResult(BaseModel):
        predicted_class: int
        probabilities: List[float]
        timestamp: str

    class ExplanationResult(BaseModel):
        predicted_class: int
        top_important_nodes: List
        attack_paths: List[Dict] = None
        timestamp: str
        node_importance_visualization: str = None
        attack_path_visualizations: List[str] = None

    # API Service
    app = FastAPI(
        title="Dynamic GNN Threat Detection API",
        description="API for detecting threats using Dynamic Graph Neural Networks",
        version="1.0.0"
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # For development
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Global objects
    CONFIG_INSTANCE = CONFIG
    data_ingestion = DataIngestion(CONFIG_INSTANCE)
    mulval_integration = MulVALIntegration(config=CONFIG_INSTANCE)
    graph_encoder = GraphEncoder(CONFIG_INSTANCE)
    data_preprocessor = DataPreprocessor(CONFIG_INSTANCE)

    # Model storage
    MODEL_STORE = {}

    @app.on_event("startup")
    async def startup_event():
        """Initialize the API on startup."""
        logger.info("Starting Dynamic GNN Threat Detection API")
        
        # Try to load a pre-trained model if available
        if HAS_TORCH:
            models_dir = Path(CONFIG_INSTANCE["models_dir"])
            checkpoints_dir = models_dir / "checkpoints"
            
            if checkpoints_dir.exists():
                best_checkpoints = list(checkpoints_dir.glob("best_*.pt"))
                if best_checkpoints:
                    try:
                        # Load the latest best checkpoint
                        best_checkpoint = sorted(best_checkpoints, key=lambda x: str(x))[-1]
                        logger.info(f"Found pre-trained model: {best_checkpoint}")
                        
                        # We'll load the model when needed
                        MODEL_STORE["best_checkpoint_path"] = str(best_checkpoint)
                    except Exception as e:
                        logger.error(f"Failed to find pre-trained model: {str(e)}")
        else:
            logger.warning("PyTorch not available. Model loading and inference disabled.")

    @app.get("/")
    async def root():
        """Root endpoint."""
        return {"message": "Dynamic GNN Threat Detection API", "status": "running", "version": "1.0.0"}

    @app.post("/api/attack-graph/generate")
    async def generate_attack_graph(
        vulnerabilities: List[VulnerabilityInput],
        network_config: NetworkConfigInput
    ):
        """
        Generate attack graph from vulnerabilities and network configuration.
        """
        try:
            logger.info("Generating attack graph")
            
            # Convert to DataFrame and dictionary
            vulnerabilities_df = pd.DataFrame([v.dict() for v in vulnerabilities])
            network_config_dict = {
                "hosts": [h.dict() for h in network_config.hosts]
            }
            
            # Prepare MulVAL input
            input_file = mulval_integration.prepare_mulval_input(vulnerabilities_df, network_config_dict)
            
            # Run MulVAL
            output_files = mulval_integration.run_mulval(input_file)
            
            # Parse attack graph
            attack_graph = mulval_integration.parse_attack_graph(output_files["attack_graph"])
            
            # Identify critical nodes
            critical_nodes = mulval_integration.identify_critical_nodes(attack_graph)
            
            # Encode graph for ML
            encoded_graph = graph_encoder.encode_attack_graph(attack_graph)
            
            # Convert NetworkX graph to node and edge lists for JSON response
            nodes = []
            for node, data in attack_graph.nodes(data=True):
                nodes.append({
                    "id": node,
                    "fact": data.get("fact", f"Node {node}"),
                    "type": data.get("type", "UNKNOWN"),
                    "critical": node in critical_nodes
                })
            
            edges = []
            for u, v in attack_graph.edges():
                edges.append({
                    "source": u,
                    "target": v
                })
            
            return {
                "nodes": nodes,
                "edges": edges,
                "critical_nodes": critical_nodes,
                "output_files": output_files
            }
        except Exception as e:
            logger.error(f"Error generating attack graph: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error generating attack graph: {str(e)}")

    @app.post("/api/detection/predict")
    async def predict_threat(
        attack_graph_file: str = None,
        network_flow_file: str = None,
        background_tasks: BackgroundTasks = None
    ):
        """
        Predict threat based on attack graph and network flow data.
        """
        try:
            logger.info("Predicting threat")
            
            # Check that we have at least one input
            if not attack_graph_file and not network_flow_file:
                raise HTTPException(status_code=400, detail="Either attack_graph_file or network_flow_file must be provided")
            
            # Load or create necessary data
            if attack_graph_file:
                # Parse attack graph
                attack_graph = mulval_integration.parse_attack_graph(attack_graph_file)
                encoded_attack_graph = graph_encoder.encode_attack_graph(attack_graph)
            else:
                # Create a placeholder
                encoded_attack_graph = None
            
            if network_flow_file:
                # Ingest network flow data
                network_flow_df = data_ingestion.ingest_network_flow(network_flow_file)
                encoded_network_flow = graph_encoder.encode_network_data(network_flow_df)
            else:
                # Create a placeholder
                encoded_network_flow = None
            
            # Combine graphs if both are available
            if encoded_attack_graph and encoded_network_flow:
                combined_graph = graph_encoder.combine_graphs(encoded_attack_graph, encoded_network_flow)
                graph_data = combined_graph
            elif encoded_attack_graph:
                graph_data = encoded_attack_graph
            else:
                graph_data = encoded_network_flow
            
            # Preprocess graph data
            preprocessed_data = data_preprocessor.preprocess_graph_encoding(graph_data)
            
            if HAS_TORCH:
                # Load model if not already loaded
                if "model" not in MODEL_STORE and "best_checkpoint_path" in MODEL_STORE:
                    # Determine model architecture from checkpoint
                    checkpoint = torch.load(MODEL_STORE["best_checkpoint_path"], map_location="cpu")
                    architecture = checkpoint.get("architecture", CONFIG_INSTANCE["model_config"]["architecture"])
                    
                    # Create model based on architecture
                    if architecture == "DynamicGNN":
                        input_dim = preprocessed_data["node_features"].shape[1]
                        model = DynamicGNN(
                            input_dim=input_dim,
                            hidden_channels=CONFIG_INSTANCE["model_config"]["hidden_channels"],
                            num_classes=2,  # Binary classification for threat detection
                            num_layers=CONFIG_INSTANCE["model_config"]["num_layers"],
                            dropout=CONFIG_INSTANCE["model_config"]["dropout"]
                        )
                    elif architecture == "EvolveGCN":
                        input_dim = preprocessed_data["node_features"].shape[1]
                        model = EvolveGCN(
                            input_dim=input_dim,
                            hidden_channels=CONFIG_INSTANCE["model_config"]["hidden_channels"],
                            num_classes=2,
                            num_layers=CONFIG_INSTANCE["model_config"]["num_layers"],
                            dropout=CONFIG_INSTANCE["model_config"]["dropout"]
                        )
                    elif architecture == "DySAT":
                        input_dim = preprocessed_data["node_features"].shape[1]
                        model = DySAT(
                            input_dim=input_dim,
                            hidden_channels=CONFIG_INSTANCE["model_config"]["hidden_channels"],
                            num_classes=2,
                            num_layers=CONFIG_INSTANCE["model_config"]["num_layers"],
                            dropout=CONFIG_INSTANCE["model_config"]["dropout"]
                        )
                    else:
                        raise ValueError(f"Unsupported model architecture: {architecture}")
                    
                    # Load model weights
                    model.load_state_dict(checkpoint["model_state_dict"])
                    MODEL_STORE["model"] = model
                    
                    logger.info(f"Loaded {architecture} model from checkpoint")
                
                # If we have a model, make a prediction
                if "model" in MODEL_STORE:
                    inference = ModelInference(MODEL_STORE["model"], CONFIG_INSTANCE)
                    prediction = inference.predict(preprocessed_data)
                    
                    # Add explanation in the background
                    if background_tasks is not None and attack_graph_file:
                        # We'll generate explanation in the background
                        explainer = ExplainabilityModule(MODEL_STORE["model"], CONFIG_INSTANCE)
                        background_tasks.add_task(
                            explainer.visualize_explanation, 
                            preprocessed_data, 
                            attack_graph
                        )
                    
                    return DetectionResult(**prediction)
            
            # If no model is available or PyTorch is not installed, return a simulated result
            logger.warning("No trained model available or PyTorch missing, returning simulated result")
            
            # Simple heuristic: if there are critical nodes, likely a threat
            if attack_graph_file:
                attack_graph = mulval_integration.parse_attack_graph(attack_graph_file)
                critical_nodes = mulval_integration.identify_critical_nodes(attack_graph)
                
                if critical_nodes:
                    predicted_class = 1  # Threat
                    probabilities = [0.3, 0.7]
                else:
                    predicted_class = 0  # No threat
                    probabilities = [0.8, 0.2]
            else:
                # Random prediction for network flow only
                predicted_class = random.randint(0, 1)
                if predicted_class == 1:
                    probabilities = [0.4, 0.6]
                else:
                    probabilities = [0.7, 0.3]
            
            return DetectionResult(
                predicted_class=predicted_class,
                probabilities=probabilities,
                timestamp=datetime.datetime.now().isoformat()
            )
                
        except Exception as e:
            logger.error(f"Error predicting threat: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error predicting threat: {str(e)}")

    # Add more API endpoints here...

    def run_api_server():
        """Run the API server."""
        host = CONFIG_INSTANCE.get("api_host", "0.0.0.0")
        port = CONFIG_INSTANCE.get("api_port", 8000)
        
        logger.info(f"Starting API server on {host}:{port}")
        uvicorn.run(app, host=host, port=port)
else:
    def run_api_server():
        """Dummy function when FastAPI is not available."""
        logger.error("FastAPI not available. Cannot run API server.")

#######################
# MAIN ENTRY POINT
#######################

def main():
    """Main entry point for the application."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Dynamic GNN Threat Detection System")
    parser.add_argument("--mode", type=str, default="demo", 
                       choices=["api", "train", "predict", "demo", "gns3"],
                       help="Operation mode: api, train, predict, demo, or gns3")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--data-dir", type=str, help="Path to data directory")
    parser.add_argument("--models-dir", type=str, help="Path to models directory")
    parser.add_argument("--mulval-path", type=str, help="Path to MulVAL installation")
    parser.add_argument("--attack-graph", type=str, help="Path to attack graph file (for predict mode)")
    parser.add_argument("--network-flow", type=str, help="Path to network flow file (for predict mode)")
    parser.add_argument("--train-data", type=str, help="Path to training dataset (for train mode)")
    
    args = parser.parse_args()
    
    # Load custom configuration if provided
    global CONFIG
    if args.config:
        try:
            with open(args.config, 'r') as f:
                custom_config = json.load(f)
                CONFIG.update(custom_config)
                logger.info(f"Loaded custom configuration from {args.config}")
        except Exception as e:
            logger.error(f"Error loading custom configuration: {str(e)}")
    
    # Override configuration with command line arguments
    if args.data_dir:
        CONFIG["data_dir"] = args.data_dir
    if args.models_dir:
        CONFIG["models_dir"] = args.models_dir
    if args.mulval_path:
        CONFIG["mulval_path"] = args.mulval_path
    
    # Ensure directories exist
    for dir_path in [CONFIG["data_dir"], CONFIG["models_dir"], CONFIG["mulval_path"]]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Run in specified mode
    if args.mode == "api":
        if HAS_FASTAPI:
            # Start API server
            run_api_server()
        else:
            logger.error("FastAPI not installed. Cannot run in API mode.")
            sys.exit(1)
    
    elif args.mode == "train":
        if not HAS_TORCH:
            logger.error("PyTorch not installed. Cannot run in train mode.")
            sys.exit(1)
            
        # Training mode
        logger.info("Running in training mode")
        
        if args.train_data:
            # Load training dataset
            data_preprocessor = DataPreprocessor(CONFIG)
            dataset = data_preprocessor.load_dataset(args.train_data)
            
            # Create DataLoaders
            train_dataset = GraphDataset(dataset['train']['encodings'], dataset['train']['labels'])
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            
            val_dataset = GraphDataset(dataset['validation']['encodings'], dataset['validation']['labels'])
            val_loader = DataLoader(val_dataset, batch_size=32)
            
            # Get input dimension from first encoding
            input_dim = dataset['train']['encodings'][0]['node_features'].shape[1]
            
            # Create model
            architecture = CONFIG["model_config"]["architecture"]
            if architecture == "DynamicGNN":
                model = DynamicGNN(
                    input_dim=input_dim,
                    hidden_channels=CONFIG["model_config"]["hidden_channels"],
                    num_classes=2,
                    num_layers=CONFIG["model_config"]["num_layers"],
                    dropout=CONFIG["model_config"]["dropout"]
                )
            elif architecture == "EvolveGCN":
                model = EvolveGCN(
                    input_dim=input_dim,
                    hidden_channels=CONFIG["model_config"]["hidden_channels"],
                    num_classes=2,
                    num_layers=CONFIG["model_config"]["num_layers"],
                    dropout=CONFIG["model_config"]["dropout"]
                )
            elif architecture == "DySAT":
                model = DySAT(
                    input_dim=input_dim,
                    hidden_channels=CONFIG["model_config"]["hidden_channels"],
                    num_classes=2,
                    num_layers=CONFIG["model_config"]["num_layers"],
                    dropout=CONFIG["model_config"]["dropout"]
                )
            else:
                logger.error(f"Unsupported model architecture: {architecture}")
                return
            
            # Create trainer
            trainer = ModelTrainer(model, CONFIG)
            
            # Train model
            logger.info(f"Starting model training with {len(train_dataset)} training samples")
            history = trainer.train(train_loader, val_loader)
            
            logger.info(f"Model training completed. Final validation accuracy: {history['val_acc'][-1]:.2f}%")
        else:
            logger.error("No training data provided. Use --train-data argument.")
    
    elif args.mode == "predict":
        if not HAS_TORCH:
            logger.error("PyTorch not installed. Cannot run in predict mode.")
            sys.exit(1)
            
        # Prediction mode
        logger.info("Running in prediction mode")
        
        if not args.attack_graph and not args.network_flow:
            logger.error("Either --attack-graph or --network-flow must be provided for predict mode")
            return
        
        # Initialize components
        data_ingestion = DataIngestion(CONFIG)
        mulval_integration = MulVALIntegration(config=CONFIG)
        graph_encoder = GraphEncoder(CONFIG)
        data_preprocessor = DataPreprocessor(CONFIG)
        
        # Load or parse data
        attack_graph = None
        encoded_attack_graph = None
        if args.attack_graph:
            attack_graph = mulval_integration.parse_attack_graph(args.attack_graph)
            encoded_attack_graph = graph_encoder.encode_attack_graph(attack_graph)
        
        network_flow_df = None
        encoded_network_flow = None
        if args.network_flow:
            network_flow_df = data_ingestion.ingest_network_flow(args.network_flow)
            encoded_network_flow = graph_encoder.encode_network_data(network_flow_df)
        
        # Combine data if both are available
        if encoded_attack_graph and encoded_network_flow:
            combined_graph = graph_encoder.combine_graphs(encoded_attack_graph, encoded_network_flow)
            graph_data = combined_graph
        elif encoded_attack_graph:
            graph_data = encoded_attack_graph
        else:
            graph_data = encoded_network_flow
        
        # Preprocess data
        preprocessed_data = data_preprocessor.preprocess_graph_encoding(graph_data)
        
        # Load model if available
        models_dir = Path(CONFIG["models_dir"])
        checkpoints_dir = models_dir / "checkpoints"
        
        if checkpoints_dir.exists():
            best_checkpoints = list(checkpoints_dir.glob("best_*.pt"))
            if best_checkpoints:
                try:
                    # Load the latest best checkpoint
                    best_checkpoint = sorted(best_checkpoints, key=lambda x: str(x))[-1]
                    logger.info(f"Found pre-trained model: {best_checkpoint}")
                    
                    # Load checkpoint
                    checkpoint = torch.load(best_checkpoint, map_location="cpu")
                    
                    # Determine model architecture
                    architecture = checkpoint.get("architecture", CONFIG["model_config"]["architecture"])
                    
                    # Create model
                    input_dim = preprocessed_data["node_features"].shape[1]
                    if architecture == "DynamicGNN":
                        model = DynamicGNN(
                            input_dim=input_dim,
                            hidden_channels=CONFIG["model_config"]["hidden_channels"],
                            num_classes=2,
                            num_layers=CONFIG["model_config"]["num_layers"],
                            dropout=CONFIG["model_config"]["dropout"]
                        )
                    elif architecture == "EvolveGCN":
                        model = EvolveGCN(
                            input_dim=input_dim,
                            hidden_channels=CONFIG["model_config"]["hidden_channels"],
                            num_classes=2,
                            num_layers=CONFIG["model_config"]["num_layers"],
                            dropout=CONFIG["model_config"]["dropout"]
                        )
                    elif architecture == "DySAT":
                        model = DySAT(
                            input_dim=input_dim,
                            hidden_channels=CONFIG["model_config"]["hidden_channels"],
                            num_classes=2,
                            num_layers=CONFIG["model_config"]["num_layers"],
                            dropout=CONFIG["model_config"]["dropout"]
                        )
                    else:
                        raise ValueError(f"Unsupported model architecture: {architecture}")
                    
                    # Load model weights
                    model.load_state_dict(checkpoint["model_state_dict"])
                    
                    # Create inference
                    inference = ModelInference(model, CONFIG)
                    
                    # Make prediction
                    prediction = inference.predict(preprocessed_data)
                    
                    # Print prediction
                    logger.info("Prediction results:")
                    logger.info(f"Predicted class: {prediction['predicted_class']}")
                    logger.info(f"Probabilities: {prediction['probabilities']}")
                    
                    # Generate explanation if attack graph is available
                    if attack_graph:
                        explainer = ExplainabilityModule(model, CONFIG)
                        explanation = explainer.explain_prediction(preprocessed_data)
                        
                        logger.info("Explanation:")
                        logger.info(f"Top important nodes: {explanation['top_important_nodes']}")
                        
                        # Generate visualizations
                        vis_info = explainer.visualize_explanation(preprocessed_data, attack_graph)
                        logger.info(f"Visualizations saved to: {vis_info['node_importance_visualization']}")
                        
                except Exception as e:
                    logger.error(f"Error loading or using pre-trained model: {str(e)}")
            else:
                logger.error("No pre-trained model found")
        else:
            logger.error("No checkpoints directory found")
    
    elif args.mode == "gns3":
        # GNS3 simulation mode
        logger.info("Running in GNS3 simulation mode")
        
        # Initialize GNS3 integration
        gns3_integration = GNS3Integration(CONFIG)
        
        # Run GNS3 simulation
        simulation_results = gns3_integration.run_gns3_simulation()
        
        if simulation_results:
            logger.info("GNS3 simulation completed successfully")
            
            # If we have network flow data, try to predict threats
            if "network_flow" in simulation_results:
                network_flow_file = simulation_results["network_flow"]
                
                if HAS_TORCH:
                    try:
                        # Initialize components
                        data_ingestion = DataIngestion(CONFIG)
                        graph_encoder = GraphEncoder(CONFIG)
                        data_preprocessor = DataPreprocessor(CONFIG)
                        
                        # Load network flow data
                        network_flow_df = data_ingestion.ingest_network_flow(network_flow_file)
                        # Encode network flow data
                        encoded_network_flow = graph_encoder.encode_network_data(network_flow_df)
                        
                        # Preprocess data
                        preprocessed_data = data_preprocessor.preprocess_graph_encoding(encoded_network_flow)
                        
                        # Load model if available
                        models_dir = Path(CONFIG["models_dir"])
                        checkpoints_dir = models_dir / "checkpoints"
                        
                        if checkpoints_dir.exists():
                            best_checkpoints = list(checkpoints_dir.glob("best_*.pt"))
                            if best_checkpoints:
                                # Load the latest best checkpoint
                                best_checkpoint = sorted(best_checkpoints, key=lambda x: str(x))[-1]
                                logger.info(f"Found pre-trained model: {best_checkpoint}")
                                
                                # Load checkpoint
                                checkpoint = torch.load(best_checkpoint, map_location="cpu")
                                
                                # Determine model architecture
                                architecture = checkpoint.get("architecture", CONFIG["model_config"]["architecture"])
                                
                                # Create model
                                input_dim = preprocessed_data["node_features"].shape[1]
                                if architecture == "DynamicGNN":
                                    model = DynamicGNN(
                                        input_dim=input_dim,
                                        hidden_channels=CONFIG["model_config"]["hidden_channels"],
                                        num_classes=2,
                                        num_layers=CONFIG["model_config"]["num_layers"],
                                        dropout=CONFIG["model_config"]["dropout"]
                                    )
                                elif architecture == "EvolveGCN":
                                    model = EvolveGCN(
                                        input_dim=input_dim,
                                        hidden_channels=CONFIG["model_config"]["hidden_channels"],
                                        num_classes=2,
                                        num_layers=CONFIG["model_config"]["num_layers"],
                                        dropout=CONFIG["model_config"]["dropout"]
                                    )
                                elif architecture == "DySAT":
                                    model = DySAT(
                                        input_dim=input_dim,
                                        hidden_channels=CONFIG["model_config"]["hidden_channels"],
                                        num_classes=2,
                                        num_layers=CONFIG["model_config"]["num_layers"],
                                        dropout=CONFIG["model_config"]["dropout"]
                                    )
                                else:
                                    raise ValueError(f"Unsupported model architecture: {architecture}")
                                
                                # Load model weights
                                model.load_state_dict(checkpoint["model_state_dict"])
                                
                                # Create inference
                                inference = ModelInference(model, CONFIG)
                                
                                # Make prediction
                                prediction = inference.predict(preprocessed_data)
                                
                                # Print prediction
                                logger.info("Prediction results:")
                                logger.info(f"Predicted class: {prediction['predicted_class']}")
                                logger.info(f"Probabilities: {prediction['probabilities']}")
                                
                                if prediction['predicted_class'] == 1:
                                    logger.warning("THREAT DETECTED in GNS3 network traffic!")
                                else:
                                    logger.info("No threats detected in GNS3 network traffic.")
                    except Exception as e:
                        logger.error(f"Error running threat detection on GNS3 data: {str(e)}")
                else:
                    logger.warning("PyTorch not available. Cannot perform threat detection on GNS3 data.")
        else:
            logger.error("GNS3 simulation failed or did not produce valid data")
    
    elif args.mode == "demo":
        # Demo mode - generate synthetic data and run through the pipeline
        logger.info("Running in demo mode")
        
        # Initialize components
        data_ingestion = DataIngestion(CONFIG)
        mulval_integration = MulVALIntegration(config=CONFIG)
        graph_encoder = GraphEncoder(CONFIG)
        data_preprocessor = DataPreprocessor(CONFIG)
        
        # Step 1: Generate synthetic data
        logger.info("Generating synthetic data")
        
        # Create synthetic vulnerabilities
        vulnerabilities = data_ingestion._generate_simulated_vulnerabilities(10)
        
        # Create synthetic network configuration
        network_config = data_ingestion._generate_simulated_network_config()
        
        # Save the synthetic data
        vulns_file = data_ingestion.save_data(vulnerabilities, "vulnerabilities", "demo_vulnerabilities")
        config_file = data_ingestion.save_data(network_config, "config", "demo_network_config")
        
        logger.info(f"Generated {len(vulnerabilities)} vulnerabilities and network configuration with {len(network_config['hosts'])} hosts")
        
        # Step 2: Generate attack graph
        logger.info("Generating attack graph")
        input_file = mulval_integration.prepare_mulval_input(vulnerabilities, network_config)
        output_files = mulval_integration.run_mulval(input_file)
        attack_graph = mulval_integration.parse_attack_graph(output_files["attack_graph"])
        
        # Step 3: Identify critical nodes
        logger.info("Identifying critical nodes")
        critical_nodes = mulval_integration.identify_critical_nodes(attack_graph)
        logger.info(f"Critical nodes: {critical_nodes}")
        
        # Step 4: Encode attack graph
        logger.info("Encoding attack graph")
        encoded_attack_graph = graph_encoder.encode_attack_graph(attack_graph)
        
        # Step 5: Generate synthetic network flows
        logger.info("Generating synthetic network flows")
        flows = data_ingestion._generate_simulated_network_data(100)
        flows_file = data_ingestion.save_data(flows, "network_flow", "demo_flows")
        
        # Step 6: Encode network flows
        logger.info("Encoding network flows")
        encoded_network_flow = graph_encoder.encode_network_data(flows)
        
        # Step 7: Combine graph encodings
        logger.info("Combining graph encodings")
        combined_graph = graph_encoder.combine_graphs(encoded_attack_graph, encoded_network_flow)
        
        # Step 8: Preprocess data
        logger.info("Preprocessing data")
        preprocessed_data = data_preprocessor.preprocess_graph_encoding(combined_graph)
        
        # Step 9: Create synthetic training data
        logger.info("Creating synthetic training dataset")
        
        # Create 20 synthetic graph samples (10 with threats, 10 without)
        graph_encodings = []
        labels = []
        
        for i in range(20):
            # Copy the base graph
            synthetic_graph = copy.deepcopy(combined_graph)
            
            # Modify node features randomly
            noise = np.random.normal(0, 0.1, synthetic_graph["node_features"].shape)
            synthetic_graph["node_features"] += noise
            
            # Add to dataset
            graph_encodings.append(synthetic_graph)
            
            # Label based on sample index (first 10 are threats, rest are not)
            labels.append(1 if i < 10 else 0)
        
        # Step 10: Create dataset
        logger.info("Creating dataset")
        dataset = data_preprocessor.create_training_dataset(graph_encodings, labels)
        
        # Step 11: Save dataset
        logger.info("Saving dataset")
        dataset_path = data_preprocessor.save_dataset(dataset, "demo_dataset")
        
        if HAS_TORCH:
            # Step 12: Create DataLoaders
            logger.info("Creating DataLoaders")
            train_dataset = GraphDataset(dataset['train']['encodings'], dataset['train']['labels'])
            train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
            
            val_dataset = GraphDataset(dataset['validation']['encodings'], dataset['validation']['labels'])
            val_loader = DataLoader(val_dataset, batch_size=4)
            
            # Step 13: Create model
            logger.info("Creating model")
            input_dim = dataset['train']['encodings'][0]['node_features'].shape[1]
            model = DynamicGNN(
                input_dim=input_dim,
                hidden_channels=CONFIG["model_config"]["hidden_channels"],
                num_classes=2,
                num_layers=CONFIG["model_config"]["num_layers"],
                dropout=CONFIG["model_config"]["dropout"]
            )
            
            # Step 14: Train model (for just a few epochs)
            logger.info("Training model")
            trainer = ModelTrainer(model, CONFIG)
            history = trainer.train(train_loader, val_loader, epochs=10)
            
            logger.info(f"Model training completed. Final validation accuracy: {history['val_acc'][-1]:.2f}%")
            
            # Step 15: Make prediction on original data
            logger.info("Making prediction")
            inference = ModelInference(model, CONFIG)
            prediction = inference.predict(preprocessed_data)
            
            logger.info("Prediction results:")
            logger.info(f"Predicted class: {prediction['predicted_class']}")
            logger.info(f"Probabilities: {prediction['probabilities']}")
            
            # Step 16: Generate explanation
            logger.info("Generating explanation")
            explainer = ExplainabilityModule(model, CONFIG)
            explanation = explainer.explain_prediction(preprocessed_data)
            
            logger.info("Explanation:")
            logger.info(f"Top important nodes: {explanation['top_important_nodes']}")
            
            # Step 17: Generate visualizations
            logger.info("Generating visualizations")
            vis_info = explainer.visualize_explanation(preprocessed_data, attack_graph)
            
            logger.info("Demo completed successfully!")
            logger.info(f"Visualizations saved to: {vis_info['node_importance_visualization']}")
            logger.info(f"Dataset saved to: {dataset_path}")
            logger.info(f"Model saved to: {trainer.save_checkpoint(is_best=True)}")
        else:
            logger.warning("PyTorch not available. Model training and inference steps skipped.")
            logger.info("Demo data generation completed successfully!")
            logger.info(f"Dataset saved to: {dataset_path}")

if __name__ == "__main__":
    import argparse
    import copy
    main()
