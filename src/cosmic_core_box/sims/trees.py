"""
Merger tree data loading for cosmological simulations.

This module provides classes for loading and processing merger tree data
from different simulation formats, with automatic field mapping and unit conversion.
"""

import numpy as np
import h5py
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from abc import ABC, abstractmethod

from .config import SimInfo


class TreeLoaderBase(ABC):
    """
    Abstract base class for merger tree data loaders.
    
    This class defines the interface that all tree loaders should implement,
    including field mapping and unit conversion functionality.
    """
    
    def __init__(self, sim_info: SimInfo):
        """
        Initialize tree loader with simulation information.
        
        Parameters:
            sim_info: SimInfo object containing simulation metadata
        """
        self.sim_info = sim_info
        self.field_mapping = self._get_field_mapping()
        self.unit_conversions = self._get_unit_conversions()
        
    @abstractmethod
    def _get_field_mapping(self) -> Dict[str, str]:
        """
        Get mapping from standard field names to simulation-specific names.
        
        Returns:
            Dictionary mapping standard names to simulation field names
        """
        pass
        
    @abstractmethod
    def _get_unit_conversions(self) -> Dict[str, float]:
        """
        Get unit conversion factors for different fields.
        
        Returns:
            Dictionary mapping field names to conversion factors
        """
        pass
        
    @abstractmethod
    def load_tree_data(self, **kwargs) -> Dict[str, np.ndarray]:
        """
        Load merger tree data.
        
        Returns:
            Dictionary containing tree data arrays
        """
        pass


class TreeLoaderTngDark(TreeLoaderBase):
    """
    Merger tree loader for IllustrisTNG dark matter only simulations.
    
    This class handles loading SubLink merger tree data from TNG format,
    with automatic field mapping and unit conversions.
    
    Attributes:
        sim_info (SimInfo): Simulation configuration object
        field_mapping (Dict[str, str]): Mapping from standard to TNG field names
        unit_conversions (Dict[str, float]): Unit conversion factors
    """
    
    def _get_field_mapping(self) -> Dict[str, str]:
        """
        Get field mapping for TNG dark matter simulations.
        
        Maps standard cosmological field names to TNG SubLink field names.
        
        Returns:
            Dictionary mapping standard names to TNG field names
        """
        return {
            'mass': 'Group_M_Crit200',
            'position': 'Group_CM',
            'velocity': 'Group_Vel',
            'halo_id': 'GroupNumber', 
            'subhalo_id': 'SubhaloNumber',
            'snapshot': 'SnapNum',
            'descendant_id': 'DescendantID',
            'first_progenitor_id': 'FirstProgenitorID',
            'next_progenitor_id': 'NextProgenitorID',
            'main_leaf_progenitor_id': 'MainLeafProgenitorID',
            'mass_peak': 'Group_M_Crit200_Peak',
            'scale_factor_peak': 'SnapNum_Peak',
            'mvir': 'Group_M_Mean200',
            'rvir': 'Group_R_Mean200',
            'r200c': 'Group_R_Crit200',
            'concentration': 'Group_Concentration',
            'spin': 'Group_Spin'
        }
        
    def _get_unit_conversions(self) -> Dict[str, float]:
        """
        Get unit conversion factors for TNG fields.
        
        Converts from TNG units to standard units (Msun/h, Mpc/h, km/s).
        
        Returns:
            Dictionary mapping field names to conversion factors
        """
        h = self.sim_info.h
        return {
            'mass': 1e10 / h,  # 1e10 Msun/h -> Msun/h
            'position': 1.0 / h,  # ckpc/h -> Mpc/h  
            'velocity': 1.0,  # km/s (already correct)
            'mvir': 1e10 / h,
            'mass_peak': 1e10 / h,
            'rvir': 1.0 / h,  # ckpc/h -> Mpc/h
            'r200c': 1.0 / h,
            'spin': 1.0 / h  # (kpc/h)(km/s) -> (Mpc/h)(km/s)
        }
        
    def load_tree_data(self, 
                      fields: Optional[List[str]] = None,
                      mass_range: Optional[Tuple[float, float]] = None,
                      snapshot_range: Optional[Tuple[int, int]] = None,
                      max_halos: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Load merger tree data from TNG SubLink files.
        
        Parameters:
            fields: List of field names to load. If None, loads all available fields
            mass_range: Tuple of (min_mass, max_mass) in Msun/h for filtering
            snapshot_range: Tuple of (min_snap, max_snap) for filtering
            max_halos: Maximum number of halos to load
            
        Returns:
            Dictionary containing loaded and processed tree data
            
        Raises:
            FileNotFoundError: If tree files are not found
            KeyError: If requested field is not available
        """
        if fields is None:
            fields = ['mass', 'position', 'velocity', 'halo_id', 'snapshot']
            
        tree_path = self.sim_info.get_tree_path()
        tree_files = list(tree_path.glob('tree_*.hdf5'))
        
        if not tree_files:
            raise FileNotFoundError(f"No tree files found in {tree_path}")
            
        # Initialize output arrays
        all_data = {field: [] for field in fields}
        total_loaded = 0
        
        for tree_file in tree_files:
            if max_halos and total_loaded >= max_halos:
                break
                
            with h5py.File(tree_file, 'r') as f:
                # Load data for requested fields
                file_data = {}
                for field in fields:
                    if field not in self.field_mapping:
                        raise KeyError(f"Unknown field: {field}")
                        
                    tng_field = self.field_mapping[field]
                    if tng_field not in f:
                        raise KeyError(f"TNG field {tng_field} not found in {tree_file}")
                        
                    data = f[tng_field][:]
                    
                    # Apply unit conversion if needed
                    if field in self.unit_conversions:
                        data = data * self.unit_conversions[field]
                        
                    file_data[field] = data
                    
                # Apply filters
                mask = np.ones(len(file_data[fields[0]]), dtype=bool)
                
                if mass_range and 'mass' in file_data:
                    mass_mask = ((file_data['mass'] >= mass_range[0]) & 
                               (file_data['mass'] <= mass_range[1]))
                    mask &= mass_mask
                    
                if snapshot_range and 'snapshot' in file_data:
                    snap_mask = ((file_data['snapshot'] >= snapshot_range[0]) &
                               (file_data['snapshot'] <= snapshot_range[1]))
                    mask &= snap_mask
                    
                # Apply mask and add to output
                n_selected = np.sum(mask)
                if max_halos:
                    remaining = max_halos - total_loaded
                    if n_selected > remaining:
                        # Randomly select subset
                        indices = np.where(mask)[0]
                        selected_indices = np.random.choice(indices, remaining, replace=False)
                        mask = np.zeros(len(mask), dtype=bool)
                        mask[selected_indices] = True
                        n_selected = remaining
                        
                for field in fields:
                    all_data[field].append(file_data[field][mask])
                    
                total_loaded += n_selected
                
        # Concatenate all data
        output_data = {}
        for field in fields:
            if all_data[field]:
                output_data[field] = np.concatenate(all_data[field])
            else:
                output_data[field] = np.array([])
                
        return output_data
        
    def load_main_branch(self, 
                        root_id: int,
                        fields: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """
        Load the main progenitor branch for a specific halo.
        
        Parameters:
            root_id: Root halo ID to trace back
            fields: Fields to load for the branch
            
        Returns:
            Dictionary containing main branch data
        """
        if fields is None:
            fields = ['mass', 'snapshot', 'halo_id']
            
        # This is a simplified implementation - actual implementation would
        # need to trace through the tree structure using progenitor links
        tree_data = self.load_tree_data(fields=fields + ['first_progenitor_id'])
        
        # Find main branch (simplified - actual implementation more complex)
        main_branch_mask = (tree_data['halo_id'] == root_id)
        
        branch_data = {}
        for field in fields:
            branch_data[field] = tree_data[field][main_branch_mask]
            
        return branch_data
