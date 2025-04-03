"""
Attack resistance system for QTrust blockchain.

This module provides a comprehensive attack detection and mitigation system for the blockchain network.
It supports various attack types detection including Sybil, Eclipse, Majority, DoS, and other common
blockchain attacks. The system combines rule-based detection with trust management integration
to identify suspicious activities and automatically respond to threats.
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict

class AttackType:
    """Attack types supported for detection."""
    ECLIPSE = "eclipse"
    SYBIL = "sybil"
    MAJORITY = "majority"
    DOS = "dos"
    REPLAY = "replay"
    MALLEABILITY = "malleability"
    PVT_KEY_COMPROMISE = "private_key_compromise"
    TIMEJACKING = "timejacking"
    SMURFING = "smurfing"
    SELFISH_MINING = "selfish_mining"

class AttackResistanceSystem:
    """
    Attack resistance system for blockchain.
    
    Detects and responds to attacks using a combination 
    of hardcoded rules and machine learning.
    """
    
    def __init__(self, 
                trust_manager, 
                validator_selector = None,
                network = None,
                detection_threshold: float = 0.65,
                auto_response: bool = True,
                collect_evidence: bool = True):
        """
        Initialize the attack resistance system.
        
        Args:
            trust_manager: System's trust manager
            validator_selector: Validator selection system (optional)
            network: Blockchain network graph (optional)
            detection_threshold: Attack detection threshold
            auto_response: Automatically respond to attacks
            collect_evidence: Collect evidence about attacks
        """
        self.trust_manager = trust_manager
        self.validator_selector = validator_selector
        self.network = network
        self.detection_threshold = detection_threshold
        self.auto_response = auto_response
        self.collect_evidence = collect_evidence
        
        # System state
        self.under_attack = False
        self.active_attacks = {}
        self.attack_evidence = defaultdict(list)
        self.attack_history = []
        
        # Store custom detection rules
        self.custom_detection_rules = []
        
        # Activity statistics
        self.stats = {
            "total_scans": 0,
            "attacks_detected": 0,
            "false_positives": 0,
            "mitigations_applied": 0
        }
    
    def scan_for_attacks(self, transaction_history: List[Dict[str, Any]], 
                        network_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Scan for potential attacks.
        
        Args:
            transaction_history: Recent transaction history
            network_state: Current network state (optional)
            
        Returns:
            Dict[str, Any]: Scan results with detected attacks
        """
        # Update scan count
        self.stats["total_scans"] += 1
        
        # Default results
        result = {
            "under_attack": False,
            "new_attacks_detected": [],
            "active_attacks": self.active_attacks.copy(),
            "confidence": 0.0,
            "recommendations": []
        }
        
        # Call attack detection method from trust manager
        if hasattr(self.trust_manager, "detect_advanced_attacks"):
            attack_detection = self.trust_manager.detect_advanced_attacks(transaction_history)
            
            # Update results from detection
            if attack_detection["under_attack"]:
                result["under_attack"] = True
                result["confidence"] = max(result["confidence"], attack_detection["confidence"])
                
                # Add information about attacks
                for attack_type in attack_detection["attack_types"]:
                    if attack_type not in self.active_attacks:
                        result["new_attacks_detected"].append(attack_type)
                        
                    # Update or add to active_attacks
                    self.active_attacks[attack_type] = {
                        "first_detected": time.time() if attack_type not in self.active_attacks else 
                                        self.active_attacks[attack_type]["first_detected"],
                        "last_detected": time.time(),
                        "confidence": attack_detection["confidence"],
                        "suspect_nodes": attack_detection["suspect_nodes"],
                        "recommended_actions": attack_detection.get("recommended_actions", [])
                    }
                
                # Update recommended actions
                result["recommendations"].extend(attack_detection.get("recommended_actions", []))
                
                # Update number of detected attacks
                if result["new_attacks_detected"]:
                    self.stats["attacks_detected"] += len(result["new_attacks_detected"])
                
                # Collect evidence if enabled
                if self.collect_evidence:
                    for attack_type in attack_detection["attack_types"]:
                        evidence = {
                            "time": time.time(),
                            "attack_type": attack_type,
                            "confidence": attack_detection["confidence"],
                            "suspect_nodes": attack_detection["suspect_nodes"].copy(),
                            "transaction_sample": transaction_history[:5] if transaction_history else []
                        }
                        self.attack_evidence[attack_type].append(evidence)
        
        # Additional checks if network state is available
        if network_state and self.network:
            # Check for Sybil attack
            sybil_confidence = self._check_for_sybil_attack(network_state)
            
            if sybil_confidence > self.detection_threshold:
                attack_type = AttackType.SYBIL
                result["under_attack"] = True
                result["confidence"] = max(result["confidence"], sybil_confidence)
                
                if attack_type not in self.active_attacks:
                    result["new_attacks_detected"].append(attack_type)
                
                self.active_attacks[attack_type] = {
                    "first_detected": time.time() if attack_type not in self.active_attacks else 
                                    self.active_attacks[attack_type]["first_detected"],
                    "last_detected": time.time(),
                    "confidence": sybil_confidence,
                    "suspect_nodes": self._identify_sybil_suspects(network_state),
                    "recommended_actions": [
                        "Increase connection threshold for new nodes",
                        "Apply more stringent ID verification",
                        "Limit connections from similar IPs",
                        "Adjust trust scores for suspicious nodes"
                    ]
                }
                
                # Update recommended actions
                result["recommendations"].extend(self.active_attacks[attack_type]["recommended_actions"])
            
            # Check for Eclipse attack
            eclipse_confidence = self._check_for_eclipse_attack(network_state)
            
            if eclipse_confidence > self.detection_threshold:
                attack_type = AttackType.ECLIPSE
                result["under_attack"] = True
                result["confidence"] = max(result["confidence"], eclipse_confidence)
                
                if attack_type not in self.active_attacks:
                    result["new_attacks_detected"].append(attack_type)
                
                self.active_attacks[attack_type] = {
                    "first_detected": time.time() if attack_type not in self.active_attacks else 
                                    self.active_attacks[attack_type]["first_detected"],
                    "last_detected": time.time(),
                    "confidence": eclipse_confidence,
                    "suspect_nodes": self._identify_eclipse_suspects(network_state),
                    "recommended_actions": [
                        "Expand mesh connections between nodes",
                        "Add trusted seed nodes",
                        "Apply connection lifecycle changes"
                    ]
                }
                
                # Update recommended actions
                result["recommendations"].extend(self.active_attacks[attack_type]["recommended_actions"])
        
        # Update system state
        self.under_attack = result["under_attack"]
        
        # Automatically respond if enabled
        if self.under_attack and self.auto_response:
            self._apply_attack_mitigations(result)
        
        return result
    
    def _check_for_sybil_attack(self, network_state: Dict[str, Any]) -> float:
        """
        Detect Sybil attack.
        
        Args:
            network_state: Current network state
            
        Returns:
            float: Detection confidence (0.0-1.0)
        """
        # Simulate Sybil detection based on network statistics
        if not network_state or not self.network:
            return 0.0
        
        # Analyze network structure
        total_nodes = len(self.network.nodes)
        if total_nodes < 5:
            return 0.0
            
        # Check nodes with abnormal connections
        suspected_count = 0
        connection_counts = {}
        
        for node in self.network.nodes:
            connection_counts[node] = len(list(self.network.neighbors(node)))
        
        # Calculate mean and standard deviation
        avg_connections = np.mean(list(connection_counts.values()))
        std_connections = np.std(list(connection_counts.values()))
        
        # Nodes with too many connections might be controlling multiple Sybils
        for node, count in connection_counts.items():
            if count > avg_connections + 2 * std_connections:
                suspected_count += 1
        
        # Calculate confidence based on ratio of suspicious nodes
        sybil_confidence = min(0.95, suspected_count / (total_nodes * 0.1))
        
        return sybil_confidence
    
    def _check_for_eclipse_attack(self, network_state: Dict[str, Any]) -> float:
        """
        Detect Eclipse attack.
        
        Args:
            network_state: Current network state
            
        Returns:
            float: Detection confidence (0.0-1.0)
        """
        if not network_state or not self.network:
            return 0.0
            
        # Eclipse attacks isolate nodes by surrounding them with malicious nodes
        # Check for nodes with limited connections to diverse nodes
        
        # Count connections to potentially isolated nodes
        isolated_nodes = []
        total_nodes = len(self.network.nodes)
        
        for node in self.network.nodes:
            neighbors = list(self.network.neighbors(node))
            unique_ips = network_state.get("unique_ips", {})
            
            # If we have IP information, check for diversity
            if node in unique_ips:
                ip_diversity = unique_ips[node].get("ip_diversity", 1.0)
                
                # Nodes with low IP diversity and few connections may be eclipsed
                if ip_diversity < 0.4 and len(neighbors) < total_nodes * 0.2:
                    isolated_nodes.append(node)
                    
        # Calculate confidence based on number of potentially isolated nodes
        if not isolated_nodes or total_nodes < 5:
            return 0.0
            
        eclipse_confidence = min(0.9, len(isolated_nodes) / (total_nodes * 0.1))
        return eclipse_confidence
    
    def _identify_sybil_suspects(self, network_state: Dict[str, Any]) -> List[int]:
        """
        Identify nodes likely involved in Sybil attack.
        
        Args:
            network_state: Current network state
            
        Returns:
            List[int]: List of suspect node IDs
        """
        if not network_state or not self.network:
            return []
            
        suspects = []
        connection_counts = {}
        
        # Get connection counts
        for node in self.network.nodes:
            connection_counts[node] = len(list(self.network.neighbors(node)))
        
        # Calculate statistics
        if not connection_counts:
            return []
            
        avg_connections = np.mean(list(connection_counts.values()))
        std_connections = np.std(list(connection_counts.values()))
        
        # Find suspicious nodes (too many connections)
        threshold = avg_connections + 2 * std_connections
        for node, count in connection_counts.items():
            if count > threshold:
                suspects.append(node)
                
        return suspects
    
    def _identify_eclipse_suspects(self, network_state: Dict[str, Any]) -> List[int]:
        """
        Identify nodes likely involved in Eclipse attack.
        
        Args:
            network_state: Current network state
            
        Returns:
            List[int]: List of suspect node IDs
        """
        if not network_state or not self.network:
            return []
            
        suspects = []
        
        # Get information about potentially isolated nodes
        isolated_nodes = []
        
        for node in self.network.nodes:
            neighbors = list(self.network.neighbors(node))
            unique_ips = network_state.get("unique_ips", {})
            
            # If we have IP information, check for diversity
            if node in unique_ips:
                ip_diversity = unique_ips[node].get("ip_diversity", 1.0)
                
                # Nodes with low IP diversity may be eclipsed
                if ip_diversity < 0.4:
                    isolated_nodes.append(node)
                    
                    # Add neighbors as suspects
                    suspects.extend(neighbors)
        
        # Remove duplicates
        return list(set(suspects))
    
    def _apply_attack_mitigations(self, attack_result: Dict[str, Any]):
        """
        Apply attack mitigation measures.
        
        Args:
            attack_result: Attack scan results
        """
        # Do nothing if no attack detected
        if not attack_result["under_attack"]:
            return
            
        # Mitigation measures applied
        mitigations_applied = []
        
        # 1. Enhance trust system
        if hasattr(self.trust_manager, "enhance_security_posture"):
            self.trust_manager.enhance_security_posture(attack_result)
            mitigations_applied.append("trust_system_enhanced")
        
        # 2. Adjust validator selection if available
        if self.validator_selector and hasattr(self.validator_selector, "update_security_level"):
            # Increase security level and force validator rotation
            if attack_result["confidence"] > 0.8:
                self.validator_selector.update_security_level("high")
                mitigations_applied.append("validator_security_level_increased")
            
            # Activate validator rotation
            if hasattr(self.validator_selector, "force_rotation"):
                self.validator_selector.force_rotation = True
                mitigations_applied.append("validator_rotation_forced")
        
        # 3. Mark suspicious nodes
        all_suspects = []
        for attack_info in self.active_attacks.values():
            all_suspects.extend(attack_info.get("suspect_nodes", []))
        
        # Remove duplicates
        all_suspects = list(set(all_suspects))
        
        # Record applied mitigation measures
        self.stats["mitigations_applied"] += len(mitigations_applied)
        
        # Record to history
        mitigation_record = {
            "time": time.time(),
            "attacks": list(self.active_attacks.keys()),
            "confidence": attack_result["confidence"],
            "mitigations_applied": mitigations_applied,
            "suspect_nodes_count": len(all_suspects)
        }
        
        self.attack_history.append(mitigation_record)
    
    def add_custom_detection_rule(self, rule_function):
        """
        Add custom detection rule.
        
        Args:
            rule_function: Custom checking function 
                        (receives network_state, returns (attack_type, confidence, suspects))
        """
        self.custom_detection_rules.append(rule_function)
    
    def clear_attack_history(self, older_than_hours: float = 24.0):
        """
        Clear old attack history.
        
        Args:
            older_than_hours: Number of hours to determine old history
        """
        if not self.attack_history:
            return
            
        current_time = time.time()
        cutoff_time = current_time - older_than_hours * 3600
        
        # Filter attack history
        self.attack_history = [record for record in self.attack_history 
                             if record["time"] >= cutoff_time]
        
        # Filter evidence
        for attack_type in list(self.attack_evidence.keys()):
            self.attack_evidence[attack_type] = [evidence for evidence in self.attack_evidence[attack_type]
                                               if evidence["time"] >= cutoff_time]
            
            # Remove key if no evidence left
            if not self.attack_evidence[attack_type]:
                del self.attack_evidence[attack_type]
        
        # Remove inactive attacks
        for attack_type in list(self.active_attacks.keys()):
            if self.active_attacks[attack_type]["last_detected"] < cutoff_time:
                del self.active_attacks[attack_type]
    
    def get_attack_report(self) -> Dict[str, Any]:
        """
        Generate report about attacks.
        
        Returns:
            Dict[str, Any]: Detailed attack report
        """
        report = {
            "under_attack": self.under_attack,
            "active_attacks": len(self.active_attacks),
            "attack_types": list(self.active_attacks.keys()),
            "attack_history_count": len(self.attack_history),
            "stats": self.stats.copy(),
            "current_recommendations": []
        }
        
        # Add recommendations from current attacks
        for attack_info in self.active_attacks.values():
            report["current_recommendations"].extend(attack_info.get("recommended_actions", []))
        
        # Remove duplicate recommendations
        report["current_recommendations"] = list(set(report["current_recommendations"]))
        
        return report 