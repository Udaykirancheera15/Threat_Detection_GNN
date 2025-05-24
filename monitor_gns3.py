#!/usr/bin/env python3
"""
GNS3 Network Monitor for Dynamic GNN Threat Detection
----------------------------------------------------
Continuous monitoring module for GNS3 environments that integrates with
the Dynamic GNN Threat Detection system.

This script provides automated monitoring capabilities:
- Establishes a persistent connection to GNS3 projects
- Captures packet data on specified intervals
- Processes network data into formats suitable for threat detection
- Integrates with the Dynamic GNN pipeline for threat analysis
- Provides alerting capabilities when threats are detected

Usage:
    python monitor_gns3.py --project [project_name] --interval [seconds] --duration [seconds]

Requirements:
    - GNS3fy library: pip install gns3fy
    - Requests library: pip install requests

Author: Security Research Team
"""

import os
import sys
import time
import json
import logging
import argparse
import datetime
import tempfile
import threading
import subprocess
import signal
from typing import Dict, List, Tuple, Union, Optional, Any, Set
from pathlib import Path
import warnings
import random
import uuid
import queue

# Network and API libraries
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    warnings.warn("Requests library not installed. API integration will be limited.")
    HAS_REQUESTS = False

# GNS3 integration
try:
    import gns3fy
    HAS_GNS3FY = True
except ImportError:
    warnings.warn("GNS3fy library not installed. GNS3 integration will be limited.")
    HAS_GNS3FY = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("gns3_monitor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("gns3_monitor")

# Default configuration
DEFAULT_CONFIG = {
    "gns3_server": "localhost",
    "gns3_port": 3080,
    "project_name": None,
    "node_filters": [],  # Empty list means monitor all nodes
    "link_filters": [],  # Empty list means monitor all links
    "capture_interval": 300,  # seconds
    "capture_duration": 60,  # seconds
    "data_dir": "./data/gns3_monitor",
    "threat_detection_api": "http://localhost:8000/api/detection/predict",
    "enable_alerts": True,
    "alert_methods": ["log", "console"],
    "alert_threshold": 0.7,  # Probability threshold for triggering alerts
    "use_simulated_data": False,  # For testing without GNS3
    "persistence": {
        "enabled": True,
        "db_path": "./data/gns3_monitor/history.json"
    }
}


class GNS3ConnectionManager:
    """
    Manages the connection to GNS3 server and projects.
    
    This class handles establishing, maintaining, and restoring
    connections to the GNS3 server, as well as selecting and
    managing projects within the server environment.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the GNS3 connection manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.server_url = f"http://{config['gns3_server']}:{config['gns3_port']}"
        self.project_name = config.get('project_name')
        self.server = None
        self.project = None
        self.connected = False
        
        # Initialize connection if GNS3fy is available
        if HAS_GNS3FY:
            self._initialize_connection()
        
    def _initialize_connection(self) -> bool:
        """
        Establish initial connection to GNS3 server.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            logger.info(f"Connecting to GNS3 server at {self.server_url}")
            self.server = gns3fy.Gns3Connector(self.server_url)
            
            # Test connection
            projects = self.server.get_projects()
            self.connected = True
            logger.info(f"Successfully connected to GNS3 server. Found {len(projects)} projects.")
            
            # Select project if specified
            if self.project_name:
                self._select_project(self.project_name)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to GNS3 server: {str(e)}")
            self.connected = False
            return False
    
    def _select_project(self, project_name: str) -> bool:
        """
        Select a specific GNS3 project.
        
        Args:
            project_name: Name of the project to select
            
        Returns:
            bool: True if project selected successfully, False otherwise
        """
        if not self.connected or not self.server:
            logger.error("Cannot select project: Not connected to GNS3 server")
            return False
        
        try:
            # Check if project exists
            projects = self.server.get_projects()
            project_exists = any(p.name == project_name for p in projects)
            
            if project_exists:
                # Get project by name
                self.project = gns3fy.Project(name=project_name, connector=self.server)
                self.project.get()
                logger.info(f"Selected project '{project_name}' (ID: {self.project.project_id})")
                return True
            else:
                logger.warning(f"Project '{project_name}' not found. Available projects: {[p.name for p in projects]}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to select project '{project_name}': {str(e)}")
            return False
    
    def get_available_projects(self) -> List[str]:
        """
        Get a list of available projects on the GNS3 server.
        
        Returns:
            List of project names
        """
        if not self.connected or not self.server:
            logger.warning("Not connected to GNS3 server")
            return []
        
        try:
            projects = self.server.get_projects()
            return [p.name for p in projects]
        except Exception as e:
            logger.error(f"Failed to get projects: {str(e)}")
            return []
    
    def reconnect(self) -> bool:
        """
        Attempt to reconnect to the GNS3 server.
        
        Returns:
            bool: True if reconnection successful, False otherwise
        """
        logger.info("Attempting to reconnect to GNS3 server...")
        
        if self._initialize_connection():
            logger.info("Reconnection successful")
            return True
        
        logger.error("Reconnection failed")
        return False
    
    def get_project_nodes(self) -> List[Dict]:
        """
        Get a list of nodes in the selected project.
        
        Returns:
            List of node dictionaries
        """
        if not self.connected or not self.project:
            logger.warning("No project selected")
            return []
        
        try:
            self.project.get()
            return self.project.nodes
        except Exception as e:
            logger.error(f"Failed to get project nodes: {str(e)}")
            return []
    
    def get_project_links(self) -> List[Dict]:
        """
        Get a list of links in the selected project.
        
        Returns:
            List of link dictionaries
        """
        if not self.connected or not self.project:
            logger.warning("No project selected")
            return []
        
        try:
            self.project.get()
            return self.project.links
        except Exception as e:
            logger.error(f"Failed to get project links: {str(e)}")
            return []
    
    def get_node_by_name(self, node_name: str) -> Optional[Dict]:
        """
        Get a specific node by name.
        
        Args:
            node_name: Name of the node to retrieve
            
        Returns:
            Node dictionary or None if not found
        """
        nodes = self.get_project_nodes()
        for node in nodes:
            if node.get('name') == node_name:
                return node
        return None
    
    def is_project_running(self) -> bool:
        """
        Check if the selected project is running.
        
        Returns:
            bool: True if project is running, False otherwise
        """
        if not self.connected or not self.project:
            return False
        
        try:
            self.project.get()
            return self.project.status == "opened"
        except Exception as e:
            logger.error(f"Failed to check project status: {str(e)}")
            return False
    
    def start_project(self) -> bool:
        """
        Start the selected project if it's not already running.
        
        Returns:
            bool: True if project started successfully, False otherwise
        """
        if not self.connected or not self.project:
            logger.warning("No project selected")
            return False
        
        try:
            if not self.is_project_running():
                logger.info(f"Starting project '{self.project.name}'")
                self.project.open()
                return True
            else:
                logger.info(f"Project '{self.project.name}' is already running")
                return True
        except Exception as e:
            logger.error(f"Failed to start project: {str(e)}")
            return False


class NetworkCapture:
    """
    Handles packet capture from GNS3 network links.
    
    This class provides functionality to capture network traffic 
    from GNS3 links, store the captures, and convert them to 
    formats suitable for analysis.
    """
    
    def __init__(self, gns3_manager: GNS3ConnectionManager, config: Dict):
        """
        Initialize network capture functionality.
        
        Args:
            gns3_manager: GNS3 connection manager
            config: Configuration dictionary
        """
        self.gns3_manager = gns3_manager
        self.config = config
        self.data_dir = Path(config['data_dir'])
        self.capture_dir = self.data_dir / "captures"
        
        # Create directory structure
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.capture_dir.mkdir(exist_ok=True)
        
        # Track active captures
        self.active_captures = {}  # {capture_id: capture_info}
    
    def start_capture(self, link_id: str) -> Optional[Dict]:
        """
        Start packet capture on a specific link.
        
        Args:
            link_id: ID of the link to capture
            
        Returns:
            Dictionary with capture information or None on failure
        """
        if not self.gns3_manager.connected or not self.gns3_manager.project:
            logger.warning("Cannot start capture: No project selected")
            return None
        
        # Create a unique capture filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        capture_file = f"capture_{link_id}_{timestamp}.pcap"
        
        try:
            # Start capture on the link
            logger.info(f"Starting capture on link {link_id}, saving to {capture_file}")
            project = self.gns3_manager.project
            
            capture = project.create_link_capture(link_id=link_id, capture_file_name=capture_file)
            
            capture_info = {
                "capture_id": capture["capture_id"],
                "link_id": link_id,
                "filename": capture_file,
                "start_time": datetime.datetime.now().isoformat(),
                "stop_time": None,
                "filepath": None,
                "status": "running"
            }
            
            self.active_captures[capture["capture_id"]] = capture_info
            return capture_info
            
        except Exception as e:
            logger.error(f"Failed to start capture on link {link_id}: {str(e)}")
            return None
    
    def stop_capture(self, capture_id: str) -> Optional[Dict]:
        """
        Stop an active packet capture.
        
        Args:
            capture_id: ID of the capture to stop
            
        Returns:
            Updated capture information dictionary or None on failure
        """
        if capture_id not in self.active_captures:
            logger.warning(f"Capture ID {capture_id} not found in active captures")
            return None
        
        try:
            # Stop the capture
            logger.info(f"Stopping capture {capture_id}")
            project = self.gns3_manager.project
            project.stop_capture(capture_id=capture_id)
            
            # Update capture info
            capture_info = self.active_captures[capture_id]
            capture_info["stop_time"] = datetime.datetime.now().isoformat()
            capture_info["status"] = "stopped"
            
            # In a real implementation, we would download the capture file
            # from the GNS3 server. Here we'll just create a placeholder.
            capture_path = self.capture_dir / capture_info["filename"]
            
            with open(capture_path, 'w') as f:
                f.write(f"# GNS3 packet capture: {capture_info['filename']}\n")
                f.write(f"# Generated at {datetime.datetime.now().isoformat()}\n")
                f.write(f"# This is a placeholder for actual PCAP data\n")
            
            capture_info["filepath"] = str(capture_path)
            return capture_info
            
        except Exception as e:
            logger.error(f"Failed to stop capture {capture_id}: {str(e)}")
            return None
    
    def stop_all_captures(self) -> List[Dict]:
        """
        Stop all active packet captures.
        
        Returns:
            List of updated capture information dictionaries
        """
        stopped_captures = []
        active_ids = list(self.active_captures.keys())
        
        for capture_id in active_ids:
            capture_info = self.stop_capture(capture_id)
            if capture_info:
                stopped_captures.append(capture_info)
        
        return stopped_captures
    
    def generate_simulated_capture(self, link_id: Optional[str] = None) -> Dict:
        """
        Generate a simulated packet capture for testing.
        
        Args:
            link_id: Optional link ID to associate with the capture
            
        Returns:
            Dictionary with simulated capture information
        """
        # Generate a random link ID if not provided
        if link_id is None:
            link_id = f"L{random.randint(1, 100)}"
        
        # Create a unique capture ID and filename
        capture_id = f"C{uuid.uuid4().hex[:8]}"
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"simulated_capture_{link_id}_{timestamp}.pcap"
        
        # Create capture file with simulated content
        capture_path = self.capture_dir / filename
        
        with open(capture_path, 'w') as f:
            f.write(f"# Simulated GNS3 packet capture\n")
            f.write(f"# Generated at {datetime.datetime.now().isoformat()}\n")
            f.write(f"# Link ID: {link_id}\n")
            f.write(f"# Capture ID: {capture_id}\n\n")
            
            # Add some simulated packet data
            for i in range(100):
                src_ip = f"192.168.1.{random.randint(1, 254)}"
                dst_ip = f"192.168.1.{random.randint(1, 254)}"
                src_port = random.randint(1024, 65535)
                dst_port = random.choice([80, 443, 22, 53, 3389, 8080])
                protocol = random.choice(["TCP", "UDP", "ICMP"])
                size = random.randint(64, 1500)
                
                f.write(f"Packet {i+1}: {src_ip}:{src_port} -> {dst_ip}:{dst_port} [{protocol}] {size} bytes\n")
        
        # Create capture info dictionary
        capture_info = {
            "capture_id": capture_id,
            "link_id": link_id,
            "filename": filename,
            "start_time": (datetime.datetime.now() - datetime.timedelta(minutes=5)).isoformat(),
            "stop_time": datetime.datetime.now().isoformat(),
            "filepath": str(capture_path),
            "status": "stopped",
            "simulated": True
        }
        
        logger.info(f"Generated simulated capture: {filename}")
        return capture_info


class ThreatDetectionIntegration:
    """
    Integrates with the Dynamic GNN Threat Detection system.
    
    This class handles sending captured network data to the 
    threat detection system and processing the results.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize threat detection integration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.api_url = config.get('threat_detection_api')
        self.alert_threshold = config.get('alert_threshold', 0.7)
        self.enable_alerts = config.get('enable_alerts', True)
        self.alert_methods = config.get('alert_methods', ["log", "console"])
        
        # Create alert history
        self.alert_history = []
        
        # Persistence setup
        self.persistence = config.get('persistence', {'enabled': True})
        if self.persistence.get('enabled', True):
            self.db_path = Path(self.persistence.get('db_path', './data/gns3_monitor/history.json'))
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._load_history()
    
    def _load_history(self):
        """Load alert history from persistence database."""
        if not self.persistence.get('enabled', True):
            return
        
        try:
            if self.db_path.exists():
                with open(self.db_path, 'r') as f:
                    data = json.load(f)
                    self.alert_history = data.get('alerts', [])
                    logger.info(f"Loaded {len(self.alert_history)} historical alerts")
        except Exception as e:
            logger.error(f"Failed to load alert history: {str(e)}")
            self.alert_history = []
    
    def _save_history(self):
        """Save alert history to persistence database."""
        if not self.persistence.get('enabled', True):
            return
        
        try:
            with open(self.db_path, 'w') as f:
                json.dump({'alerts': self.alert_history}, f, indent=2)
            logger.debug(f"Saved {len(self.alert_history)} alerts to history")
        except Exception as e:
            logger.error(f"Failed to save alert history: {str(e)}")
    
    def analyze_capture(self, capture_info: Dict) -> Dict:
        """
        Send a capture file to the threat detection system for analysis.
        
        Args:
            capture_info: Capture information dictionary
            
        Returns:
            Dictionary with analysis results
        """
        if not HAS_REQUESTS:
            return self._simulate_analysis(capture_info)
        
        filepath = capture_info.get("filepath")
        if not filepath or not Path(filepath).exists():
            logger.warning(f"Capture file not found: {filepath}")
            return self._simulate_analysis(capture_info)
        
        if not self.api_url:
            logger.warning("No threat detection API URL configured")
            return self._simulate_analysis(capture_info)
        
        try:
            logger.info(f"Sending capture {capture_info['filename']} to threat detection API")
            
            # Prepare the file for upload
            files = {'network_flow': open(filepath, 'rb')}
            
            # Send the request
            response = requests.post(self.api_url, files=files)
            
            # Check response
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Analysis complete: Threat probability {result.get('probabilities', [0, 0])[1]:.2f}")
                
                # Augment with capture info
                result['capture_info'] = capture_info
                result['timestamp'] = datetime.datetime.now().isoformat()
                
                # Process alerts if needed
                if self.is_threat_detected(result):
                    self._process_alert(result)
                
                return result
            else:
                logger.error(f"API request failed: {response.status_code} {response.text}")
                return self._simulate_analysis(capture_info)
                
        except Exception as e:
            logger.error(f"Failed to analyze capture: {str(e)}")
            return self._simulate_analysis(capture_info)
    
    def _simulate_analysis(self, capture_info: Dict) -> Dict:
        """
        Generate simulated analysis results for testing.
        
        Args:
            capture_info: Capture information dictionary
            
        Returns:
            Dictionary with simulated analysis results
        """
        logger.info(f"Generating simulated analysis for {capture_info.get('filename', 'unknown')}")
        
        # Generate a random threat probability
        threat_prob = random.uniform(0, 1)
        
        # Determine prediction based on probability
        predicted_class = 1 if threat_prob > 0.5 else 0
        
        # Create result dictionary
        result = {
            "predicted_class": predicted_class,
            "probabilities": [1 - threat_prob, threat_prob],
            "timestamp": datetime.datetime.now().isoformat(),
            "capture_info": capture_info,
            "simulated": True
        }
        
        # Process alerts if needed
        if self.is_threat_detected(result):
            self._process_alert(result)
        
        return result
    
    def is_threat_detected(self, result: Dict) -> bool:
        """
        Determine if a threat is detected based on analysis result.
        
        Args:
            result: Analysis result dictionary
            
        Returns:
            bool: True if threat is detected, False otherwise
        """
        # Check if predicted class is 1 (threat) or if probability exceeds threshold
        if result.get("predicted_class") == 1:
            return True
        
        probabilities = result.get("probabilities", [1.0, 0.0])
        if len(probabilities) > 1 and probabilities[1] >= self.alert_threshold:
            return True
        
        return False
    
    def _process_alert(self, result: Dict):
        """
        Process a threat alert.
        
        Args:
            result: Analysis result dictionary
        """
        if not self.enable_alerts:
            return
        
        # Create alert message
        capture_info = result.get("capture_info", {})
        filename = capture_info.get("filename", "unknown")
        threat_prob = result.get("probabilities", [0, 0])[1]
        
        alert_msg = f"THREAT DETECTED in {filename} with {threat_prob:.2f} probability!"
        
        # Add to alert history
        alert_record = {
            "timestamp": datetime.datetime.now().isoformat(),
            "message": alert_msg,
            "result": result,
            "acknowledged": False
        }
        self.alert_history.append(alert_record)
        
        # Save history
        self._save_history()
        
        # Process alerts based on configured methods
        if "log" in self.alert_methods:
            logger.warning(alert_msg)
        
        if "console" in self.alert_methods:
            print(f"\n\033[91m*** ALERT: {alert_msg} ***\033[0m\n")
        
        if "file" in self.alert_methods:
            alert_dir = Path(self.config.get('data_dir')) / "alerts"
            alert_dir.mkdir(exist_ok=True)
            
            alert_file = alert_dir / f"alert_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(alert_file, 'w') as f:
                json.dump(alert_record, f, indent=2)
    
    def get_recent_alerts(self, limit: int = 10) -> List[Dict]:
        """
        Get recent threat alerts.
        
        Args:
            limit: Maximum number of alerts to return
            
        Returns:
            List of alert dictionaries
        """
        # Return most recent alerts first
        return sorted(self.alert_history, key=lambda x: x.get("timestamp", ""), reverse=True)[:limit]
    
    def acknowledge_alert(self, alert_timestamp: str) -> bool:
        """
        Mark an alert as acknowledged.
        
        Args:
            alert_timestamp: Timestamp of the alert to acknowledge
            
        Returns:
            bool: True if alert acknowledged, False otherwise
        """
        for alert in self.alert_history:
            if alert.get("timestamp") == alert_timestamp:
                alert["acknowledged"] = True
                self._save_history()
                return True
        
        return False


class MonitoringScheduler:
    """
    Schedules and manages continuous monitoring tasks.
    
    This class handles scheduled execution of network captures
    and threat detection analysis based on configured intervals.
    """
    
    def __init__(self, gns3_manager: GNS3ConnectionManager, 
                network_capture: NetworkCapture,
                threat_detection: ThreatDetectionIntegration,
                config: Dict):
        """
        Initialize the monitoring scheduler.
        
        Args:
            gns3_manager: GNS3 connection manager
            network_capture: Network capture handler
            threat_detection: Threat detection integration
            config: Configuration dictionary
        """
        self.gns3_manager = gns3_manager
        self.network_capture = network_capture
        self.threat_detection = threat_detection
        self.config = config
        
        self.capture_interval = config.get('capture_interval', 300)  # seconds
        self.capture_duration = config.get('capture_duration', 60)  # seconds
        self.node_filters = config.get('node_filters', [])
        self.link_filters = config.get('link_filters', [])
        
        # Store monitoring state
        self.running = False
        self.last_capture_time = None
        self.scheduled_stop = None
        self.monitoring_thread = None
        self.stop_event = threading.Event()
        
        # Queue for monitoring results
        self.result_queue = queue.Queue()
    
    def filter_links(self, links: List[Dict]) -> List[Dict]:
        """
        Filter links based on configured filters.
        
        Args:
            links: List of link dictionaries
            
        Returns:
            Filtered list of links
        """
        if not self.link_filters:
            return links
        
        filtered_links = []
        for link in links:
            # Check if link matches any filter
            link_id = link.get("link_id", "")
            nodes = []
            
            # Extract nodes connected by this link
            for node in link.get("nodes", []):
                if "label" in node and "name" in node["label"]:
                    nodes.append(node["label"]["name"])
            
            # Check if any node in the link matches a filter
            if any(node_name in self.node_filters for node_name in nodes):
                filtered_links.append(link)
                continue
            
            # Check if link ID matches a filter
            if any(filter_str in link_id for filter_str in self.link_filters):
                filtered_links.append(link)
        
        return filtered_links
    
    def start_monitoring(self):
        """Start continuous monitoring."""
        if self.running:
            logger.warning("Monitoring is already running")
            return
        
        self.running = True
        self.stop_event.clear()
        
        # Start monitoring in a separate thread
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        logger.info(f"Started continuous monitoring (interval: {self.capture_interval}s, "
                   f"duration: {self.capture_duration}s)")
    
    def stop_monitoring(self):
        """Stop continuous monitoring."""
        if not self.running:
            logger.warning("Monitoring is not running")
            return
        
        logger.info("Stopping monitoring...")
        self.running = False
        self.stop_event.set()
        
        # Stop any active captures
        self.network_capture.stop_all_captures()
        
        # Wait for monitoring thread to complete
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=10)
        
        logger.info("Monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop executed in a separate thread."""
        while self.running and not self.stop_event.is_set():
            try:
                # Capture cycle
                logger.info("Starting capture cycle")
                
                # Get links to monitor
                if self.config.get('use_simulated_data', False):
                    # Use simulated data
                    active_captures = []
                    for i in range(3):  # Simulate 3 links
                        capture_info = self.network_capture.generate_simulated_capture()
                        active_captures.append(capture_info)
                else:
                    # Get actual links from GNS3
                    if not self.gns3_manager.connected:
                        logger.warning("Not connected to GNS3. Attempting to reconnect...")
                        self.gns3_manager.reconnect()
                    
                    if not self.gns3_manager.connected:
                        logger.error("Failed to connect to GNS3. Using simulated data.")
                        # Use simulated data as fallback
                        capture_info = self.network_capture.generate_simulated_capture()
                        active_captures = [capture_info]
                    else:
                        # Get links from GNS3
                        links = self.gns3_manager.get_project_links()
                        filtered_links = self.filter_links(links)
                        
                        if not filtered_links:
                            logger.warning("No links match the filters. Using all links.")
                            filtered_links = links
                        
                        # Start captures
                        active_captures = []
                        for link in filtered_links:
                            link_id = link.get("link_id")
                            capture_info = self.network_capture.start_capture(link_id)
                            if capture_info:
                                active_captures.append(capture_info)
                
                if not active_captures:
                    logger.warning("No active captures started")
                else:
                    logger.info(f"Started {len(active_captures)} captures")
                    # Wait for capture duration
                logger.info(f"Capturing for {self.capture_duration} seconds...")
                time.sleep(self.capture_duration)
                
                # Stop captures
                completed_captures = []
                for capture_info in active_captures:
                    capture_id = capture_info.get("capture_id")
                    if "simulated" in capture_info and capture_info["simulated"]:
                        # No need to stop simulated captures
                        completed_captures.append(capture_info)
                    else:
                        # Stop actual GNS3 capture
                        stopped_info = self.network_capture.stop_capture(capture_id)
                        if stopped_info:
                            completed_captures.append(stopped_info)
                
                # Analyze captures
                for capture_info in completed_captures:
                    logger.info(f"Analyzing capture: {capture_info.get('filename')}")
                    result = self.threat_detection.analyze_capture(capture_info)
                    
                    # Add result to queue for external access
                    self.result_queue.put(result)
                    
                    # Log result
                    threat_prob = result.get("probabilities", [0, 0])[1]
                    logger.info(f"Analysis result: Threat probability {threat_prob:.2f}")
                
                # Wait until next interval
                self.last_capture_time = datetime.datetime.now()
                next_capture = self.last_capture_time + datetime.timedelta(seconds=self.capture_interval)
                
                logger.info(f"Capture cycle completed. Next capture scheduled at {next_capture.strftime('%H:%M:%S')}")
                
                # Wait for next interval (but check for stop event every second)
                wait_time = self.capture_interval - self.capture_duration
                if wait_time > 0:
                    for _ in range(wait_time):
                        if self.stop_event.is_set():
                            break
                        time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(30)  # Wait a bit before retrying
    
    def get_status(self) -> Dict:
        """
        Get current monitoring status.
        
        Returns:
            Dictionary with monitoring status
        """
        next_capture = None
        if self.last_capture_time:
            next_capture = self.last_capture_time + datetime.timedelta(seconds=self.capture_interval)
        
        return {
            "running": self.running,
            "last_capture_time": self.last_capture_time.isoformat() if self.last_capture_time else None,
            "next_capture_time": next_capture.isoformat() if next_capture else None,
            "capture_interval": self.capture_interval,
            "capture_duration": self.capture_duration,
            "node_filters": self.node_filters,
            "link_filters": self.link_filters
        }
    
    def get_latest_results(self, limit: int = 5) -> List[Dict]:
        """
        Get latest monitoring results.
        
        Args:
            limit: Maximum number of results to return
            
        Returns:
            List of result dictionaries
        """
        # Collect results from queue without blocking
        results = []
        for _ in range(min(self.result_queue.qsize(), limit)):
            try:
                results.append(self.result_queue.get_nowait())
            except queue.Empty:
                break
        
        return results


class MonitoringController:
    """
    Central controller for GNS3 monitoring functionality.
    
    This class serves as the main interface for the monitoring system,
    coordinating the GNS3 connection, network captures, threat detection,
    and scheduling components.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the monitoring controller.
        
        Args:
            config: Configuration dictionary
        """
        # Use default config if none provided
        self.config = config or DEFAULT_CONFIG
        
        # Create data directory
        data_dir = Path(self.config.get('data_dir', './data/gns3_monitor'))
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.gns3_manager = GNS3ConnectionManager(self.config)
        self.network_capture = NetworkCapture(self.gns3_manager, self.config)
        self.threat_detection = ThreatDetectionIntegration(self.config)
        self.scheduler = MonitoringScheduler(
            self.gns3_manager,
            self.network_capture,
            self.threat_detection,
            self.config
        )
        
        logger.info("GNS3 Monitor initialized")
    
    def start_monitoring(self):
        """Start continuous monitoring."""
        self.scheduler.start_monitoring()
    
    def stop_monitoring(self):
        """Stop continuous monitoring."""
        self.scheduler.stop_monitoring()
    
    def get_monitoring_status(self) -> Dict:
        """
        Get current monitoring status.
        
        Returns:
            Dictionary with monitoring status
        """
        return self.scheduler.get_status()
    
    def get_available_projects(self) -> List[str]:
        """
        Get list of available GNS3 projects.
        
        Returns:
            List of project names
        """
        return self.gns3_manager.get_available_projects()
    
    def select_project(self, project_name: str) -> bool:
        """
        Select a specific GNS3 project.
        
        Args:
            project_name: Name of the project to select
            
        Returns:
            bool: True if project selected successfully, False otherwise
        """
        result = self.gns3_manager._select_project(project_name)
        if result:
            self.config['project_name'] = project_name
        return result
    
    def get_nodes(self) -> List[Dict]:
        """
        Get nodes in the current project.
        
        Returns:
            List of node dictionaries
        """
        return self.gns3_manager.get_project_nodes()
    
    def get_links(self) -> List[Dict]:
        """
        Get links in the current project.
        
        Returns:
            List of link dictionaries
        """
        return self.gns3_manager.get_project_links()
    
    def get_recent_alerts(self, limit: int = 10) -> List[Dict]:
        """
        Get recent threat alerts.
        
        Args:
            limit: Maximum number of alerts to return
            
        Returns:
            List of alert dictionaries
        """
        return self.threat_detection.get_recent_alerts(limit)
    
    def acknowledge_alert(self, alert_timestamp: str) -> bool:
        """
        Mark an alert as acknowledged.
        
        Args:
            alert_timestamp: Timestamp of the alert to acknowledge
            
        Returns:
            bool: True if alert acknowledged, False otherwise
        """
        return self.threat_detection.acknowledge_alert(alert_timestamp)
    
    def get_latest_results(self, limit: int = 5) -> List[Dict]:
        """
        Get latest monitoring results.
        
        Args:
            limit: Maximum number of results to return
            
        Returns:
            List of result dictionaries
        """
        return self.scheduler.get_latest_results(limit)
    
    def update_config(self, new_config: Dict) -> bool:
        """
        Update monitoring configuration.
        
        Args:
            new_config: New configuration values
            
        Returns:
            bool: True if config updated successfully, False otherwise
        """
        try:
            # Store current monitoring state
            monitoring_active = self.scheduler.running
            
            # Stop monitoring if active
            if monitoring_active:
                self.stop_monitoring()
            
            # Update config values
            for key, value in new_config.items():
                if key in self.config:
                    self.config[key] = value
            
            # Reinitialize components if necessary
            if 'capture_interval' in new_config or 'capture_duration' in new_config:
                self.scheduler.capture_interval = self.config.get('capture_interval', 300)
                self.scheduler.capture_duration = self.config.get('capture_duration', 60)
            
            if 'node_filters' in new_config or 'link_filters' in new_config:
                self.scheduler.node_filters = self.config.get('node_filters', [])
                self.scheduler.link_filters = self.config.get('link_filters', [])
            
            if 'gns3_server' in new_config or 'gns3_port' in new_config:
                # Reconnect to GNS3
                self.gns3_manager = GNS3ConnectionManager(self.config)
                self.network_capture = NetworkCapture(self.gns3_manager, self.config)
                self.scheduler.gns3_manager = self.gns3_manager
                self.scheduler.network_capture = self.network_capture
            
            # Restart monitoring if it was active
            if monitoring_active:
                self.start_monitoring()
            
            logger.info("Configuration updated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update configuration: {str(e)}")
            return False
    
    def run_manual_capture(self, link_id: str = None, duration: int = None) -> Dict:
        """
        Run a manual capture on a specific link.
        
        Args:
            link_id: ID of the link to capture (use simulated if None)
            duration: Duration in seconds (use config value if None)
            
        Returns:
            Dictionary with capture results
        """
        if duration is None:
            duration = self.config.get('capture_duration', 60)
        
        # Check if we should use simulated data
        if link_id is None or self.config.get('use_simulated_data', False):
            # Use simulated data
            capture_info = self.network_capture.generate_simulated_capture(link_id)
        else:
            # Start capture
            capture_info = self.network_capture.start_capture(link_id)
            if not capture_info:
                logger.warning(f"Failed to start capture on link {link_id}. Using simulated data.")
                capture_info = self.network_capture.generate_simulated_capture(link_id)
            else:
                # Wait for specified duration
                logger.info(f"Capturing for {duration} seconds...")
                time.sleep(duration)
                
                # Stop capture
                capture_info = self.network_capture.stop_capture(capture_info["capture_id"])
        
        # Analyze capture
        if capture_info:
            logger.info(f"Analyzing capture: {capture_info.get('filename')}")
            result = self.threat_detection.analyze_capture(capture_info)
            
            # Add result to queue for external access
            self.scheduler.result_queue.put(result)
            
            return result
        else:
            logger.error("Failed to create or complete capture")
            return {
                "error": "Failed to create or complete capture",
                "timestamp": datetime.datetime.now().isoformat()
            }


def signal_handler(sig, frame):
    """Handle termination signals gracefully."""
    logger.info("Termination signal received. Shutting down...")
    
    # Check if we have a controller instance
    if 'controller' in globals():
        controller.stop_monitoring()
    
    sys.exit(0)


def interactive_mode(controller: MonitoringController):
    """
    Run in interactive console mode.
    
    Args:
        controller: Monitoring controller instance
    """
    print("\nGNS3 Network Monitor - Interactive Mode")
    print("=======================================")
    
    while True:
        print("\nCommand options:")
        print("  1. Start monitoring")
        print("  2. Stop monitoring")
        print("  3. Show status")
        print("  4. List available projects")
        print("  5. Select project")
        print("  6. Show nodes and links")
        print("  7. Show recent alerts")
        print("  8. Run manual capture")
        print("  9. Show latest results")
        print("  0. Exit")
        
        choice = input("\nEnter command number: ")
        
        if choice == "1":
            controller.start_monitoring()
            print("Monitoring started")
        
        elif choice == "2":
            controller.stop_monitoring()
            print("Monitoring stopped")
        
        elif choice == "3":
            status = controller.get_monitoring_status()
            print("\nMonitoring Status:")
            for key, value in status.items():
                print(f"  {key}: {value}")
        
        elif choice == "4":
            projects = controller.get_available_projects()
            print("\nAvailable Projects:")
            for i, proj in enumerate(projects, 1):
                print(f"  {i}. {proj}")
            
            if not projects:
                print("  No projects found or not connected to GNS3")
        
        elif choice == "5":
            projects = controller.get_available_projects()
            if not projects:
                print("No projects available")
                continue
            
            print("\nAvailable Projects:")
            for i, proj in enumerate(projects, 1):
                print(f"  {i}. {proj}")
            
            idx = input("Enter project number to select: ")
            try:
                idx = int(idx) - 1
                if 0 <= idx < len(projects):
                    result = controller.select_project(projects[idx])
                    if result:
                        print(f"Project '{projects[idx]}' selected successfully")
                    else:
                        print(f"Failed to select project '{projects[idx]}'")
                else:
                    print("Invalid project number")
            except ValueError:
                print("Invalid input")
        
        elif choice == "6":
            nodes = controller.get_nodes()
            links = controller.get_links()
            
            print("\nNodes:")
            for i, node in enumerate(nodes, 1):
                print(f"  {i}. {node.get('name', 'Unknown')} (ID: {node.get('node_id', 'Unknown')})")
            
            if not nodes:
                print("  No nodes found or no project selected")
            
            print("\nLinks:")
            for i, link in enumerate(links, 1):
                link_id = link.get('link_id', 'Unknown')
                nodes_info = []
                for node in link.get('nodes', []):
                    if 'label' in node and 'name' in node['label']:
                        nodes_info.append(node['label']['name'])
                    else:
                        nodes_info.append("Unknown node")
                
                print(f"  {i}. Link {link_id}: {' <-> '.join(nodes_info)}")
            
            if not links:
                print("  No links found or no project selected")
        
        elif choice == "7":
            alerts = controller.get_recent_alerts()
            print("\nRecent Alerts:")
            for i, alert in enumerate(alerts, 1):
                timestamp = alert.get('timestamp', 'Unknown')
                message = alert.get('message', 'Unknown alert')
                ack = "Acknowledged" if alert.get('acknowledged', False) else "Not acknowledged"
                print(f"  {i}. [{timestamp}] {message} ({ack})")
            
            if not alerts:
                print("  No alerts found")
        
        elif choice == "8":
            links = controller.get_links()
            if not links and not controller.config.get('use_simulated_data', False):
                print("No links found or no project selected")
                continue
            
            if controller.config.get('use_simulated_data', False) or not links:
                print("Using simulated link data")
                result = controller.run_manual_capture()
            else:
                print("\nAvailable Links:")
                for i, link in enumerate(links, 1):
                    link_id = link.get('link_id', 'Unknown')
                    nodes_info = []
                    for node in link.get('nodes', []):
                        if 'label' in node and 'name' in node['label']:
                            nodes_info.append(node['label']['name'])
                        else:
                            nodes_info.append("Unknown node")
                    
                    print(f"  {i}. Link {link_id}: {' <-> '.join(nodes_info)}")
                
                idx = input("Enter link number to capture (0 for simulated): ")
                try:
                    idx = int(idx)
                    if idx == 0:
                        result = controller.run_manual_capture()
                    elif 1 <= idx <= len(links):
                        link_id = links[idx-1].get('link_id')
                        result = controller.run_manual_capture(link_id)
                    else:
                        print("Invalid link number")
                        continue
                except ValueError:
                    print("Invalid input")
                    continue
            
            threat_prob = result.get("probabilities", [0, 0])[1]
            print(f"\nCapture analysis complete:")
            print(f"  Threat probability: {threat_prob:.2f}")
            print(f"  Predicted class: {result.get('predicted_class', 'Unknown')}")
            
            if controller.threat_detection.is_threat_detected(result):
                print("  \033[91mTHREAT DETECTED!\033[0m")
        
        elif choice == "9":
            results = controller.get_latest_results()
            print("\nLatest Results:")
            for i, result in enumerate(results, 1):
                threat_prob = result.get("probabilities", [0, 0])[1]
                timestamp = result.get("timestamp", "Unknown")
                capture_info = result.get("capture_info", {})
                filename = capture_info.get("filename", "Unknown")
                
                print(f"  {i}. [{timestamp}] {filename}: Threat probability {threat_prob:.2f}")
                if controller.threat_detection.is_threat_detected(result):
                    print("     \033[91mTHREAT DETECTED!\033[0m")
            
            if not results:
                print("  No results found")
        
        elif choice == "0":
            print("Exiting...")
            controller.stop_monitoring()
            break
        
        else:
            print("Invalid command")


def main():
    """Main entry point for the script."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="GNS3 Network Monitor for Dynamic GNN Threat Detection")
    parser.add_argument("--project", type=str, help="GNS3 project name to monitor")
    parser.add_argument("--server", type=str, default="localhost", help="GNS3 server address")
    parser.add_argument("--port", type=int, default=3080, help="GNS3 server port")
    parser.add_argument("--interval", type=int, default=300, help="Capture interval in seconds")
    parser.add_argument("--duration", type=int, default=60, help="Capture duration in seconds")
    parser.add_argument("--data-dir", type=str, help="Data directory path")
    parser.add_argument("--api-url", type=str, help="Threat detection API URL")
    parser.add_argument("--simulate", action="store_true", help="Use simulated data instead of actual GNS3 data")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    
    args = parser.parse_args()
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Load configuration
    config = DEFAULT_CONFIG.copy()
    
    # Load from file if provided
    if args.config:
        try:
            with open(args.config, 'r') as f:
                file_config = json.load(f)
                config.update(file_config)
            logger.info(f"Loaded configuration from {args.config}")
        except Exception as e:
            logger.error(f"Failed to load configuration from {args.config}: {str(e)}")
    
    # Update with command line arguments
    if args.project:
        config["project_name"] = args.project
    if args.server:
        config["gns3_server"] = args.server
    if args.port:
        config["gns3_port"] = args.port
    if args.interval:
        config["capture_interval"] = args.interval
    if args.duration:
        config["capture_duration"] = args.duration
    if args.data_dir:
        config["data_dir"] = args.data_dir
    if args.api_url:
        config["threat_detection_api"] = args.api_url
    if args.simulate:
        config["use_simulated_data"] = True
    
    # Create and initialize controller
    global controller
    controller = MonitoringController(config)
    
    # Run in interactive or daemon mode
    if args.interactive:
        interactive_mode(controller)
    else:
        logger.info("Starting monitoring in daemon mode")
        controller.start_monitoring()
        
        try:
            # Keep main thread alive
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        finally:
            controller.stop_monitoring()
            logger.info("Monitoring stopped")
    
    logger.info("GNS3 Monitor exiting")


if __name__ == "__main__":
    main()
