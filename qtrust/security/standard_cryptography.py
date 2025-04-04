"""
Standard Cryptography Module for QTrust

Provides standard cryptographic operations for the QTrust system.
"""

import hashlib
import hmac
import os
from enum import Enum, auto
from typing import Dict, Tuple, Optional, Union, ByteString

class KeyType(Enum):
    """Types of cryptographic keys supported."""
    SYMMETRIC = auto()
    ASYMMETRIC_PRIVATE = auto()
    ASYMMETRIC_PUBLIC = auto()

class HashAlgorithm(Enum):
    """Hash algorithms supported by the system."""
    SHA256 = auto()
    SHA512 = auto()
    BLAKE2B = auto()
    KECCAK256 = auto()

class StandardCryptography:
    """
    Provides standard cryptographic operations for blockchain security.
    This includes hashing, digital signatures, and key management.
    """
    
    def __init__(self, default_hash_algorithm: HashAlgorithm = HashAlgorithm.SHA256):
        """
        Initialize the cryptography module.
        
        Args:
            default_hash_algorithm: Default hashing algorithm to use
        """
        self.default_hash_algorithm = default_hash_algorithm
        self._key_store: Dict[str, Tuple[KeyType, bytes]] = {}
    
    def hash_data(self, data: Union[str, bytes], algorithm: Optional[HashAlgorithm] = None) -> bytes:
        """
        Hash data using the specified algorithm.
        
        Args:
            data: Data to hash (string or bytes)
            algorithm: Hashing algorithm to use (defaults to instance default)
            
        Returns:
            bytes: Hash digest
        """
        if algorithm is None:
            algorithm = self.default_hash_algorithm
            
        if isinstance(data, str):
            data = data.encode('utf-8')
            
        if algorithm == HashAlgorithm.SHA256:
            return hashlib.sha256(data).digest()
        elif algorithm == HashAlgorithm.SHA512:
            return hashlib.sha512(data).digest()
        elif algorithm == HashAlgorithm.BLAKE2B:
            return hashlib.blake2b(data).digest()
        elif algorithm == HashAlgorithm.KECCAK256:
            # Note: This is a simple approximation for keccak256
            # In production, you would use a proper Keccak implementation
            return hashlib.sha3_256(data).digest()
        else:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")
    
    def generate_key(self, key_type: KeyType, key_id: str) -> str:
        """
        Generate a new cryptographic key and store it.
        
        Args:
            key_type: Type of key to generate
            key_id: Identifier for the key
            
        Returns:
            str: The key ID
        """
        if key_type == KeyType.SYMMETRIC:
            key_material = os.urandom(32)  # 256-bit key
        elif key_type == KeyType.ASYMMETRIC_PRIVATE:
            # In a real implementation, this would use proper asymmetric key generation
            # This is a placeholder
            key_material = os.urandom(64)
        else:
            raise ValueError(f"Cannot directly generate key of type {key_type}")
            
        self._key_store[key_id] = (key_type, key_material)
        return key_id
    
    def sign_data(self, data: Union[str, bytes], key_id: str) -> bytes:
        """
        Create a digital signature for the data.
        
        Args:
            data: Data to sign
            key_id: ID of the key to use for signing
            
        Returns:
            bytes: Digital signature
        """
        if key_id not in self._key_store:
            raise ValueError(f"Key {key_id} not found")
            
        key_type, key_material = self._key_store[key_id]
        
        if key_type != KeyType.SYMMETRIC and key_type != KeyType.ASYMMETRIC_PRIVATE:
            raise ValueError(f"Cannot sign with key type {key_type}")
            
        if isinstance(data, str):
            data = data.encode('utf-8')
            
        # Simple HMAC-based signature for demonstration
        # In production, would use proper digital signature algorithms
        return hmac.new(key_material, data, hashlib.sha256).digest()
    
    def verify_signature(self, data: Union[str, bytes], signature: bytes, key_id: str) -> bool:
        """
        Verify a digital signature.
        
        Args:
            data: The original data
            signature: The signature to verify
            key_id: ID of the key to use for verification
            
        Returns:
            bool: True if signature is valid, False otherwise
        """
        if key_id not in self._key_store:
            raise ValueError(f"Key {key_id} not found")
            
        key_type, key_material = self._key_store[key_id]
        
        if isinstance(data, str):
            data = data.encode('utf-8')
            
        # Simple HMAC verification
        expected_signature = hmac.new(key_material, data, hashlib.sha256).digest()
        return hmac.compare_digest(signature, expected_signature)
    
    def secure_hash_for_storage(self, data: Union[str, bytes]) -> str:
        """
        Create a hash suitable for secure storage (like passwords).
        
        Args:
            data: Data to hash
            
        Returns:
            str: Hex representation of the hash
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
            
        # Use a more secure algorithm for stored hashes
        return hashlib.blake2b(data, digest_size=32).hexdigest()
    
    def compute_merkle_root(self, items: list) -> bytes:
        """
        Compute a Merkle root hash from a list of items.
        
        Args:
            items: List of items to include in the Merkle tree
            
        Returns:
            bytes: Merkle root hash
        """
        if not items:
            return b"\x00" * 32  # Empty root
            
        # Convert items to bytes and hash them
        leaves = [self.hash_data(item) for item in items]
        
        # Build tree bottom-up
        while len(leaves) > 1:
            if len(leaves) % 2 != 0:
                leaves.append(leaves[-1])  # Duplicate last node if odd
                
            next_level = []
            for i in range(0, len(leaves), 2):
                combined = leaves[i] + leaves[i+1]
                next_level.append(self.hash_data(combined))
                
            leaves = next_level
            
        return leaves[0] 