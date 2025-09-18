"""
Snapshot data loading for cosmological simulations.

This module provides classes for loading particle data from simulation snapshots,
with support for chunked reading of large datasets.
"""

import numpy as np
import h5py
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Generator
from abc import ABC, abstractmethod

from .config import SimInfo


class SnapshotLoaderBase(ABC):
    """
    Abstract base class for snapshot data loaders.
    
    Defines the interface for loading particle data from simulation snapshots.
    """
    
    def __init__(self, sim_info: SimInfo):
        """
        Initialize snapshot loader with simulation information.
        
        Parameters:
            sim_info: SimInfo object containing simulation metadata
        """
        self.sim_info = sim_info
        
    @abstractmethod
    def load_particles(self, **kwargs) -> Dict[str, np.ndarray]:
        """
        Load particle data from snapshot.
        
        Returns:
            Dictionary containing particle data arrays
        """
        pass


class SnapshotLoaderTngDark(SnapshotLoaderBase):
    """
    Snapshot loader for IllustrisTNG dark matter only simulations.
    
    This class handles loading dark matter particle data from TNG snapshots,
    with support for chunked reading and spatial filtering.
    
    Attributes:
        sim_info (SimInfo): Simulation configuration object
        particle_type (int): TNG particle type for dark matter (1)
    """
    
    def __init__(self, sim_info: SimInfo):
        """
        Initialize TNG dark matter snapshot loader.
        
        Parameters:
            sim_info: SimInfo object containing simulation metadata
        """
        super().__init__(sim_info)
        self.particle_type = 1  # Dark matter particles in TNG
        
    def load_particles(self,
                        snapshot_num: int,
                        fields: Optional[List[str]] = None,
                        chunk_size: Optional[int] = None,
                        spatial_filter: Optional[Dict] = None) -> Dict[str, np.ndarray]:
        """
        Load dark matter particle data from a TNG snapshot.
        
        Parameters:
            snapshot_num: Snapshot number to load
            fields: List of particle fields to load ('pos', 'vel', 'id')
            chunk_size: Size of chunks for memory-efficient loading
            spatial_filter: Dict with 'center', 'radius' for spatial filtering
            
        Returns:
            Dictionary containing particle data arrays
            
        Raises:
            FileNotFoundError: If snapshot file is not found
            KeyError: If requested field is not available
        """
        if fields is None:
            fields = ['pos', 'vel']
            
        snapshot_path = self.sim_info.get_snapshot_path(snapshot_num)
        
        # Handle multi-file snapshots (TNG format)
        if not snapshot_path.exists():
            # Try multi-file format
            base_name = snapshot_path.stem
            snapshot_files = list(snapshot_path.parent.glob(f'{base_name}.*.hdf5'))
            if not snapshot_files:
                raise FileNotFoundError(f"Snapshot files not found: {snapshot_path}")
        else:
            snapshot_files = [snapshot_path]
            
        # Map field names to TNG dataset names
        field_mapping = {
            'pos': 'Coordinates',
            'vel': 'Velocities', 
            'id': 'ParticleIDs',
            'mass': 'Masses'
        }
        
        # Initialize output arrays
        all_data = {field: [] for field in fields}
        
        for snap_file in snapshot_files:
            with h5py.File(snap_file, 'r') as f:
                # Get particle type group
                pt_group = f[f'PartType{self.particle_type}']
                
                # Load requested fields
                file_data = {}
                for field in fields:
                    if field not in field_mapping:
                        raise KeyError(f"Unknown field: {field}")
                        
                    dataset_name = field_mapping[field]
                    if dataset_name not in pt_group:
                        raise KeyError(f"Field {dataset_name} not found in {snap_file}")
                        
                    if chunk_size:
                        # Load in chunks
                        dataset = pt_group[dataset_name]
                        chunks = []
                        for i in range(0, len(dataset), chunk_size):
                            chunk = dataset[i:i+chunk_size]
                            chunks.append(chunk)
                        data = np.concatenate(chunks)
                    else:
                        data = pt_group[dataset_name][:]
                        
                    # Apply unit conversions
                    if field == 'pos':
                        data = data / self.sim_info.h  # ckpc/h -> Mpc/h
                    elif field == 'mass':
                        data = data * 1e10 / self.sim_info.h  # 1e10 Msun/h -> Msun/h
                        
                    file_data[field] = data
                    
                # Apply spatial filter if specified
                if spatial_filter and 'pos' in file_data:
                    center = np.array(spatial_filter['center'])
                    radius = spatial_filter['radius']
                    
                    # Calculate distances (accounting for periodic boundaries)
                    pos = file_data['pos']
                    box_size = self.sim_info.box_size
                    
                    # Handle periodic boundaries
                    dx = pos - center
                    dx = np.where(dx > box_size/2, dx - box_size, dx)
                    dx = np.where(dx < -box_size/2, dx + box_size, dx)
                    
                    distances = np.linalg.norm(dx, axis=1)
                    mask = distances <= radius
                    
                    # Apply mask to all fields
                    for field in fields:
                        file_data[field] = file_data[field][mask]
                        
                # Add to output
                for field in fields:
                    all_data[field].append(file_data[field])
                    
        # Concatenate data from all files
        output_data = {}
        for field in fields:
            if all_data[field]:
                output_data[field] = np.concatenate(all_data[field])
            else:
                output_data[field] = np.array([])
                
        return output_data
        
    def load_particles_chunked(self,
                                snapshot_num: int,
                                fields: Optional[List[str]] = None,
                                chunk_size: int = 1000000) -> Generator[Dict[str, np.ndarray], None, None]:
        """
        Generator for loading particle data in chunks.
        
        This method yields chunks of particle data, useful for processing
        very large datasets that don't fit in memory.
        
        Parameters:
            snapshot_num: Snapshot number to load
            fields: List of particle fields to load
            chunk_size: Number of particles per chunk
            
        Yields:
            Dictionary containing chunk of particle data
        """
        if fields is None:
            fields = ['pos']
            
        snapshot_path = self.sim_info.get_snapshot_path(snapshot_num)
        
        # Handle multi-file snapshots
        if not snapshot_path.exists():
            base_name = snapshot_path.stem
            snapshot_files = list(snapshot_path.parent.glob(f'{base_name}.*.hdf5'))
            if not snapshot_files:
                raise FileNotFoundError(f"Snapshot files not found: {snapshot_path}")
        else:
            snapshot_files = [snapshot_path]
            
        field_mapping = {
            'pos': 'Coordinates',
            'vel': 'Velocities',
            'id': 'ParticleIDs',
            'mass': 'Masses'
        }
        
        for snap_file in snapshot_files:
            with h5py.File(snap_file, 'r') as f:
                pt_group = f[f'PartType{self.particle_type}']
                
                # Get total number of particles in this file
                n_particles = len(pt_group['Coordinates'])
                
                # Process in chunks
                for i in range(0, n_particles, chunk_size):
                    end_idx = min(i + chunk_size, n_particles)
                    
                    chunk_data = {}
                    for field in fields:
                        dataset_name = field_mapping[field]
                        data = pt_group[dataset_name][i:end_idx]
                        
                        # Apply unit conversions
                        if field == 'pos':
                            data = data / self.sim_info.h
                        elif field == 'mass':
                            data = data * 1e10 / self.sim_info.h
                            
                        chunk_data[field] = data
                        
                    yield chunk_data
                    
    def get_snapshot_info(self, snapshot_num: int) -> Dict[str, Any]:
        """
        Get metadata information about a snapshot.
        
        Parameters:
            snapshot_num: Snapshot number
            
        Returns:
            Dictionary containing snapshot metadata
        """
        snapshot_path = self.sim_info.get_snapshot_path(snapshot_num)
        
        with h5py.File(snapshot_path, 'r') as f:
            header = dict(f['Header'].attrs)
            
            info = {
                'time': header['Time'],
                'redshift': header['Redshift'],
                'box_size': header['BoxSize'] / self.sim_info.h,  # Convert to Mpc/h
                'n_particles': header['NumPart_Total'],
                'mass_table': header['MassTable'] * 1e10 / self.sim_info.h,  # Msun/h
                'omega_m': header['Omega0'],
                'omega_l': header['OmegaLambda'],
                'h': header['HubbleParam']
            }
            
        return info
