'''
Class to calculate nearby chains
'''



import pandas as pd
import numpy as np
import networkx as nx
from sklearn.neighbors import KDTree
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Any


class NearChains:
    """
    A class to cluster events that are nearby in both space and time given specific thresholds
    
    Attributes:
        df (pd.DataFrame): DataFrame containing x, y, datetime columns
        kdtree (KDTree): KDTree built from spatial coordinates
    """
    
    def __init__(self, df: pd.DataFrame, x: str, y: str, d: str):
        """
        Initialize the clustering class.
        
        Args:
            df (pd.DataFrame): DataFrame with columns 'x', 'y', 'datetime'
            x (str): string with field for x coordinate
            y (str): string with field for y coordinate
            d (str): string with field for datetime value
        """
        # needs to have no missing data
        self.df = df[~df[[x,y,d]].isna().any(axis=1)]
        self.x = x
        self.y = y
        self.d = d
        self.vars = [x,y,d]
        
        # Convert datetime column to pandas datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(self.df['datetime']):
            self.df[d] = pd.to_datetime(self.df[d])
        
        # Build KDTree from spatial coordinates
        self.spatial_coords = self.df[[x,y]].values
        self.kdtree = KDTree(self.spatial_coords)
        # in days since the first event in the sample
        self.day_second = 60*60*24
        self.timestamps = (self.df[d] - self.df[d].min()).dt.total_seconds()/self.day_second
        self.timestamps = self.timestamps.values
    
    def get_clusters(self,time_thresh,space_thresh) -> List[pd.DataFrame]:
        """
        Find connected components of events that are nearby in both space and time.
        
        Args:
            time_thresh (float): temporal period to consider two events linked (in days)
            space_thresh (float): distance to consider two events linked
        
        Returns:
            List[pd.DataFrame]: List of connected components, where each component 
                           contains the dataframe rows corresponding to the linked events
        """
        # Query all points at once for spatial neighbors
        neighbor_indices = self.kdtree.query_radius(
            self.spatial_coords, 
            r=space_thresh
        )
        
        # Collect all unique pairs (i, j) where i < j and they are spatially close
        spatial_pairs = []
        for i, spatial_neighbors in enumerate(neighbor_indices):
            # Only consider neighbors with index > i to avoid duplicates
            valid_neighbors = spatial_neighbors[spatial_neighbors > i]
            spatial_pairs.extend([(i, j) for j in valid_neighbors])
        
        if not spatial_pairs:
            print("No spatially nearby pairs found")
            # Return empty list
            return []
        
        # Convert to numpy arrays for vectorized operations
        spatial_pairs = np.array(spatial_pairs)
        i_indices = spatial_pairs[:, 0]
        j_indices = spatial_pairs[:, 1]
        
        # Vectorized time difference calculation
        time_diffs = np.abs(self.timestamps[i_indices] - self.timestamps[j_indices])
        
        # Filter pairs that are close in time
        valid_time_mask = time_diffs <= time_thresh
        valid_pairs = spatial_pairs[valid_time_mask]
        
        # Create NetworkX graph
        G = nx.Graph()
        G.add_edges_from(valid_pairs)
        
        # Find connected components
        connected_components = list(nx.connected_components(G))
        
        # Convert sets to lists and sort for consistency
        connected_components = [sorted(list(component)) for component in connected_components]
        
        # Sort components by size (largest first) and then by smallest index
        connected_components.sort(key=lambda x: (-len(x), min(x)))
        
        print(f"Found {len(connected_components)} connected components")
        print(f"Processed {len(valid_pairs)} valid spatiotemporal pairs")
        
        # return a list of the original dataframe components
        comp_df = [self.df.iloc[c].sort_values(by=self.d) for c in connected_components]
        
        return comp_df
    
    def get_component_summary(self,complist) -> pd.DataFrame:
        """
        Get a summary of connected components with statistics.
        
        Args:
            complist: list of dataframes (from get_clusters)
        
        Returns:
            pd.DataFrame: Summary with component_id, size, min/max dates, and centroid of events
        """
        summary_data = []
        for i, component in enumerate(complist):
            summary_data.append({
                'component_id': i,
                'size': component.shape[0],
                'min_datetime': component[self.d].min(),
                'max_datetime': component[self.d].max(),
                'center_x': component[self.x].mean(),
                'center_y': component[self.y].mean()
            })
        
        return pd.DataFrame(summary_data)


