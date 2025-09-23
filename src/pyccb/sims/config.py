"""
Configuration management for cosmological simulations.

This module provides the SimInfo class for loading and managing simulation
metadata from JSON configuration files.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import numpy as np


class SimInfo:
    """
    A class to load and manage simulation metadata and configuration.
    
    This class reads simulation parameters from JSON configuration files
    and provides convenient access to cosmological parameters, box size,
    redshift lists, and file paths.
    
    Attributes:
        name (str): Simulation name identifier
        config_data (Dict[str, Any]): Raw configuration data from JSON
        box_size (float): Simulation box size in Mpc/h
        cosmology (Dict[str, float]): Cosmological parameters
        redshifts (np.ndarray): Array of available redshifts
        snapshots (Dict[int, Dict]): Snapshot information
        base_path (Path): Base path for simulation data files
    """
    
    def __init__(self, config_file: Union[str, Path]):
        """
        Initialize SimInfo from a JSON configuration file.
        
        Parameters:
            config_file: Path to the JSON configuration file
            
        Raises:
            FileNotFoundError: If configuration file doesn't exist
            json.JSONDecodeError: If configuration file is not valid JSON
        """
        self.config_file = Path(config_file)
        if not self.config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
            
        with open(self.config_file, 'r') as f:
            self.config_data = json.load(f)
            
        self._parse_config()
        
    def _parse_config(self) -> None:
        """Parse configuration data and set instance attributes."""
        self.name = self.config_data.get('name', 'unknown')
        
        # Simulation parameters
        sim_params = self.config_data.get('simulation', {})
        self.box_size = sim_params.get('box_size', 75.0)  # Mpc/h
        
        # Cosmological parameters
        self.cosmology = self.config_data.get('cosmology', {})
        
        # Redshift information
        redshift_list = self.config_data.get('redshifts', [])
        self.redshifts = np.array(redshift_list)
        
        # Snapshot information
        self.snapshots = self.config_data.get('snapshots', {})
        
        # File paths
        paths = self.config_data.get('paths', {})
        self.base_path = Path(paths.get('base_path', '.'))
        self.tree_path = self.base_path / paths.get('trees', 'trees')
        self.snapshot_path = self.base_path / paths.get('snapshots', 'snapshots')
        
    @property
    def omega_m(self) -> float:
        """Matter density parameter."""
        return self.cosmology.get('Omega_m', 0.3089)
        
    @property
    def omega_l(self) -> float:
        """Dark energy density parameter."""
        return self.cosmology.get('Omega_Lambda', 0.6911)
        
    @property
    def omega_b(self) -> float:
        """Baryon density parameter."""
        return self.cosmology.get('Omega_b', 0.0486)
        
    @property
    def h(self) -> float:
        """Dimensionless Hubble parameter."""
        return self.cosmology.get('h', 0.6774)
        
    @property
    def sigma8(self) -> float:
        """RMS fluctuation amplitude at 8 Mpc/h."""
        return self.cosmology.get('sigma_8', 0.8159)
        
    @property
    def ns(self) -> float:
        """Scalar spectral index."""
        return self.cosmology.get('n_s', 0.9667)
        
    def get_snapshot_path(self, snapshot_num: int) -> Path:
        """
        Get the file path for a specific snapshot.
        
        Parameters:
            snapshot_num: Snapshot number
            
        Returns:
            Path to the snapshot file(s)
            
        Raises:
            KeyError: If snapshot number is not found in configuration
        """
        if str(snapshot_num) not in self.snapshots:
            raise KeyError(f"Snapshot {snapshot_num} not found in configuration")
            
        snap_info = self.snapshots[str(snapshot_num)]
        filename = snap_info.get('filename', f'snap_{snapshot_num:03d}.hdf5')
        return self.snapshot_path / filename
        
    def get_tree_path(self) -> Path:
        """
        Get the path to merger tree files.
        
        Returns:
            Path to merger tree files
        """
        return self.tree_path
        
    def get_redshift_from_snapshot(self, snapshot_num: int) -> float:
        """
        Get redshift corresponding to a snapshot number.
        
        Parameters:
            snapshot_num: Snapshot number
            
        Returns:
            Redshift value
            
        Raises:
            KeyError: If snapshot number is not found
        """
        if str(snapshot_num) not in self.snapshots:
            raise KeyError(f"Snapshot {snapshot_num} not found in configuration")
            
        return self.snapshots[str(snapshot_num)].get('redshift', 0.0)
        
    def get_snapshot_from_redshift(self, redshift: float, tolerance: float = 0.1) -> int:
        """
        Get snapshot number closest to a given redshift.
        
        Parameters:
            redshift: Target redshift
            tolerance: Maximum allowed difference in redshift
            
        Returns:
            Closest snapshot number
            
        Raises:
            ValueError: If no snapshot found within tolerance
        """
        min_diff = float('inf')
        closest_snap = None
        
        for snap_str, snap_info in self.snapshots.items():
            snap_z = snap_info.get('redshift', 0.0)
            diff = abs(snap_z - redshift)
            if diff < min_diff:
                min_diff = diff
                closest_snap = int(snap_str)
                
        if min_diff > tolerance:
            raise ValueError(f"No snapshot found within {tolerance} of z={redshift}")
            
        return closest_snap
        
    def __repr__(self) -> str:
        """String representation of SimInfo object."""
        return (f"SimInfo(name='{self.name}', box_size={self.box_size}, "
                f"n_snapshots={len(self.snapshots)})")
