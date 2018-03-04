import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import ipdb

def main():
    test1()

def test1():
    G = nx.DiGraph()
    G.add_edge('A', 'B')#, weight=4)
    G.add_edge('B', 'D', weight=2)
    G.add_edge('A', 'C', weight=3)
    G.add_edge('C', 'D', weight=4)
    path = nx.shortest_path(G, 'A', 'D', weight='weight')
    am = nx.adjacency_matrix(G)
    am_np = np.array(am.todense())
    print(path)
    print(am)
    nx.draw(G, with_labels=True, font_weight='bold')
    plt.show()

def test2():
    G = nx.petersen_graph()
    plt.subplot(121)
    nx.draw(G, with_labels=True, font_weight='bold')
    plt.subplot(122)
    nx.draw_shell(G, nlist=[range(5,10), range(5)], with_labels=True, font_weight='bold')
    #plt.show()

def test3():
    G = nx.DiGraph()
    G.add_edge('A', 'B', etype='gg')
    G.add_edge('B', 'D', etype='kk')
    G.add_edge('A', 'C', etype='kk')
    G.add_edge('C', 'D', etype='gg')
    for u,v,d in G.edges(data=True):
        if d['etype']=='gg':
            print(u,v,d)

if __name__=="__main__":
    main()
