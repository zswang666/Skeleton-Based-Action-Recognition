import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random

def create_adjacency_matrix(g, n_nodes, n_edge_types, label_mapping):
    '''
    Create adjacency matrix for graph with
    - undirected edges in DiGraph
    - 2 types of edges: spatial and temporal
    '''
    a = np.zeros([n_nodes, n_nodes * n_edge_types * 2])
    for u,v,d in g.edges(data=True):
        src = label_mapping[u]
        tgt = label_mapping[v]
        if d["etype"]=="spatial":
            etype = 0
        elif d["etype"]=="temporal":
            etype = 1
        a[tgt][etype*n_nodes+src] = 1
        a[src][(etype+n_edge_types)*n_nodes+tgt] = 1

    return a

### Currently cannot be used because of unresolved package mayavi
def draw_graph3d(graph, graph_colormap='winter', bgcolor = (1, 1, 1),
                 node_size=0.03,
                 edge_color=(0.8, 0.8, 0.8), edge_size=0.002,
                 text_size=0.008, text_color=(0, 0, 0)):

    H=nx.Graph()

    # add edges
    for node, edges in graph.items():
        for edge, val in edges.items():
            if val == 1:
                H.add_edge(node, edge)

    G=nx.convert_node_labels_to_integers(H)

    graph_pos=nx.spring_layout(G, dim=3)

    # numpy array of x,y,z positions in sorted node order
    xyz=np.array([graph_pos[v] for v in sorted(G)])

    # scalar colors
    scalars=np.array(G.nodes())+5
    mlab.figure(1, bgcolor=bgcolor)
    mlab.clf()

    #----------------------------------------------------------------------------
    # the x,y, and z co-ordinates are here
    # manipulate them to obtain the desired projection perspective 
    pts = mlab.points3d(xyz[:,0], xyz[:,1], xyz[:,2],
                        scalars,
                        scale_factor=node_size,
                        scale_mode='none',
                        colormap=graph_colormap,
                        resolution=20)
    #----------------------------------------------------------------------------

    for i, (x, y, z) in enumerate(xyz):
        label = mlab.text(x, y, str(i), z=z,
                          width=text_size, name=str(i), color=text_color)
        label.property.shadow = True

    pts.mlab_source.dataset.lines = np.array(G.edges())
    tube = mlab.pipeline.tube(pts, tube_radius=edge_size)
    mlab.pipeline.surface(tube, color=edge_color)

    mlab.show() # interactive window
