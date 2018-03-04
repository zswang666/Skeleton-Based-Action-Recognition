import networkx as nx
import numpy as np

def full_joint_label_mapping():
    return {
        "Nose_{}".format(t): 0+18*t,
        "Neck_{}".format(t): 1+18*t,
        "RShoulder_{}".format(t): 2+18*t,
        "RElbow_{}".format(t): 3+18*t,
        "RWrist_{}".format(t): 4,
        "LShoulder_{}".format(t): 5+18*t,
        "LElbow_{}".format(t): 6+18*t,
        "LWrist_{}".format(t): 7+18*t,
        "RHip_{}".format(t): 8+18*t,
        "RKnee_{}".format(t): 9+18*t,
        "RAnkle_{}".format(t): 10+18*t,
        "LHip_{}".format(t): 11+18*t,
        "LKnee_{}".format(t): 12+18*t,
        "LAnkle_{}".format(t): 13+18*t,
        "REye_{}".format(t): 14+18*t,
        "LEye_{}".format(t): 15+18*t,
        "REar_{}".format(t): 16+18*t,
        "LEar_{}".format(t): 17+18*t,
        "Background_{}".format(t): 18+18*t
    }

def joint_label_mapping(timesteps):
    def dict_template(t):
        return {
            "Nose_{}".format(t): 0+18*t,
            "Neck_{}".format(t): 1+18*t,
            "RShoulder_{}".format(t): 2+18*t,
            "RElbow_{}".format(t): 3+18*t,
            "RWrist_{}".format(t): 4,
            "LShoulder_{}".format(t): 5+18*t,
            "LElbow_{}".format(t): 6+18*t,
            "LWrist_{}".format(t): 7+18*t,
            "RHip_{}".format(t): 8+18*t,
            "RKnee_{}".format(t): 9+18*t,
            "RAnkle_{}".format(t): 10+18*t,
            "LHip_{}".format(t): 11+18*t,
            "LKnee_{}".format(t): 12+18*t,
            "LAnkle_{}".format(t): 13+18*t,
            "REye_{}".format(t): 14+18*t,
            "LEye_{}".format(t): 15+18*t,
            "REar_{}".format(t): 16+18*t,
            "LEar_{}".format(t): 17+18*t,
            "Background_{}".format(t): 18+18*t
        }
    lp = dict()
    for t in range(timesteps):
        lp.update(dict_template(t))

    return lp

def create_graph(timesteps):
    # edges for all joints 
    edges_prefix = []
    edges_prefix.append(("Nose","Neck"))
    edges_prefix.append(("Neck","RShoulder"))
    edges_prefix.append(("RShoulder","RElbow"))
    edges_prefix.append(("RElbow","RWrist"))
    
    edges_prefix.append(("Neck","LShoulder"))
    edges_prefix.append(("LShoulder","LElbow"))
    edges_prefix.append(("LElbow","LWrist"))
    edges_prefix.append(("Neck","RHip"))
    edges_prefix.append(("RHip","RKnee"))
    edges_prefix.append(("RKnee","RAnkle"))
    edges_prefix.append(("Neck","LHip"))
    edges_prefix.append(("LHip","LKnee"))
    edges_prefix.append(("LKnee","LAnkle"))
    edges_prefix.append(("Nose","REye"))
    edges_prefix.append(("REye","REar"))
    edges_prefix.append(("Nose","LEye"))
    edges_prefix.append(("LEye","LEar"))

    G = nx.DiGraph()
    # create spatial edges
    for t in range(timesteps):
        for ep in edges_prefix:
            node1 = ep[0] + "_{}".format(t)
            node2 = ep[1] + "_{}".format(t)
            G.add_edge(node1,node2,etype="spatial")
            G.add_edge(node2,node1,etype="spatial")
    # create temporal edges
    for t in range(timesteps-1):
        for ep in edges_prefix:
            node1 = ep[0] + "_{}".format(t)
            node1_next = ep[0] + "_{}".format(t+1)
            node2 = ep[1] + "_{}".format(t)
            node2_next = ep[1] + "_{}".format(t+1)
            G.add_edge(node1,node1_next,etype="temporal")
            G.add_edge(node1_next,node1,etype="temporal")
            G.add_edge(node2,node2_next,etype="temporal")
            G.add_edge(node2_next,node2,etype="temporal")
   
    return G
