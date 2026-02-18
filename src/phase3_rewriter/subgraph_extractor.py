"""
Subgraph Extractor

Extracts the computational subgraph that produces a specific activation.

TODO: Implement in Phase 3 (Weeks 7-9)
"""

from typing import List, Set


class SubgraphExtractor:
    """
    Extracts subgraphs from the computation graph.
    """
    
    def __init__(self, computation_graph):
        """
        Initialize extractor.
        
        Args:
            computation_graph: Full computation graph
        """
        self.graph = computation_graph
    
    def extract_subgraph_for_activation(self, tensor_id: int) -> List[int]:
        """
        Extract the subgraph that computes a specific activation.
        
        Args:
            tensor_id: ID of the activation tensor
            
        Returns:
            List of node IDs that compute this activation
        """
        # TODO: Implement subgraph extraction
        # Should trace backward from the activation to find all operations needed
        raise NotImplementedError("Phase 3: To be implemented")
    
    def find_dependencies(self, node_id: int) -> Set[int]:
        """
        Find all dependencies of a node.
        
        Args:
            node_id: Node to find dependencies for
            
        Returns:
            Set of node IDs that this node depends on
        """
        # TODO: Implement dependency tracking
        raise NotImplementedError("Phase 3: To be implemented")
