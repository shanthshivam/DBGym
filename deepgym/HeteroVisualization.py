from torch_geometric.data import HeteroData
import networkx as nx
import matplotlib.pyplot as plt

def visualize_hetero(data : HeteroData):
    '''
    Input a HeteroData of PyG and we'll draw it for you if it is not too big!
    data : the HeteroData you want to visualize.
    '''
    # Only available for small graphs.
    if data.num_nodes > 100:
        raise ValueError("This graph is too big to draw.")
    
    # Create a NetworkX MultiDiGraph
    G_nx = nx.MultiDiGraph()

    # print(data.edge_types)
    # print(data.edge_stores)
    # print()
    # Add nodes and edges for each relation
    for i, edge in zip(range(len(data.edge_types)), data.edge_stores):
        edge = edge["edge_index"]
        # print(i,edge)
        # print(data.edge_types[i])
        for j in range(edge.shape[1]):
            # Use a string to represent a node to differentiate between types.
            src = str(edge[0][j].item()) + data.edge_types[i][0]
            dst = str(edge[1][j].item()) + data.edge_types[i][2]
            # Add type information as well.
            G_nx.add_node(src, node_type = data.edge_types[i][0])
            G_nx.add_node(dst, node_type = data.edge_types[i][2])
            # print(src,dst)
            G_nx.add_edge(src, dst, key = data.edge_types[i])

    # Visualize the graph
    pos = nx.multipartite_layout(G_nx, subset_key='node_type')  # You can choose different layout algorithms
        
    # Create a colormap based on unique edge types
    edge_types = data.edge_types
    unique_edge_types = list(set(edge_types))
    edge_color_map = {edge_type: plt.cm.jet(i / len(unique_edge_types)) for i, edge_type in enumerate(unique_edge_types)}
    edge_colors = [edge_color_map[edge_type] for edge_type in edge_types]

    nx.draw(G_nx, pos, with_labels=True, edge_color=edge_colors, node_color='lightblue', node_size=1000)
    plt.show()
    return