"""
Graph manipulation utilities.
"""

import networkx as nx
from typing import Dict, List, Set, Tuple, Optional


def create_networkx_graph(computation_graph) -> nx.DiGraph:
    """
    Convert our ComputationGraph to NetworkX DiGraph for analysis.
    
    Args:
        computation_graph: Our ComputationGraph instance
        
    Returns:
        NetworkX directed graph
    """
    G = nx.DiGraph()
    
    # Add nodes
    for node_id, node in computation_graph.nodes.items():
        G.add_node(node_id, 
                  operation=node.operation_name,
                  op_type=node.operation_type,
                  forward=node.is_forward,
                  backward=node.is_backward)
    
    # Add edges based on tensor flow
    for node_id, node in computation_graph.nodes.items():
        for tensor_id in node.input_tensors:
            tensor_info = computation_graph.tensors[tensor_id]
            if tensor_info.producer_node is not None:
                G.add_edge(tensor_info.producer_node, node_id, 
                          tensor_id=tensor_id)
    
    return G


def find_longest_path(computation_graph) -> List[int]:
    """
    Find the longest path in the computation graph.
    
    Args:
        computation_graph: Our ComputationGraph instance
        
    Returns:
        List of node IDs representing the longest path
    """
    G = create_networkx_graph(computation_graph)
    
    try:
        # Find longest path using DAG longest path algorithm
        path = nx.dag_longest_path(G)
        return path
    except:
        # If graph has cycles or other issues, return execution order
        return computation_graph.execution_order


def find_critical_tensors(computation_graph, threshold_mb: float = 10.0) -> List[int]:
    """
    Find tensors that consume significant memory.
    
    Args:
        computation_graph: Our ComputationGraph instance
        threshold_mb: Minimum size in MB to be considered critical
        
    Returns:
        List of critical tensor IDs
    """
    critical = []
    threshold_bytes = threshold_mb * (1024 ** 2)
    
    for tensor_id, tensor_info in computation_graph.tensors.items():
        if tensor_info.size_bytes >= threshold_bytes:
            critical.append(tensor_id)
    
    return critical


def compute_graph_depth(computation_graph) -> int:
    """
    Compute the depth of the computation graph.
    
    Args:
        computation_graph: Our ComputationGraph instance
        
    Returns:
        Maximum depth of the graph
    """
    G = create_networkx_graph(computation_graph)
    
    try:
        # Compute longest path length
        return len(nx.dag_longest_path(G))
    except:
        return len(computation_graph.execution_order)


def get_forward_subgraph(computation_graph) -> Set[int]:
    """
    Get all nodes that are part of the forward pass.
    
    Args:
        computation_graph: Our ComputationGraph instance
        
    Returns:
        Set of forward node IDs
    """
    forward_nodes = set()
    
    for node_id, node in computation_graph.nodes.items():
        if node.is_forward:
            forward_nodes.add(node_id)
    
    return forward_nodes


def get_backward_subgraph(computation_graph) -> Set[int]:
    """
    Get all nodes that are part of the backward pass.
    
    Args:
        computation_graph: Our ComputationGraph instance
        
    Returns:
        Set of backward node IDs
    """
    backward_nodes = set()
    
    for node_id, node in computation_graph.nodes.items():
        if node.is_backward:
            backward_nodes.add(node_id)
    
    return backward_nodes
