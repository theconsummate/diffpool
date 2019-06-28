import networkx as nx
import numpy as np
import scipy as sc
import os
import re
import pickle

def read_graphfile(datadir, dataname, max_nodes=None):
    ''' Read data from https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets
        graph index starts with 1 in file

    Returns:
        List of networkx objects with graph and node labels
    '''
    prefix = os.path.join(datadir, dataname, dataname)
    filename_graph_indic = prefix + '_graph_indicator.txt'
    # index of graphs that a given node belongs to
    graph_indic={}
    with open(filename_graph_indic) as f:
        i=1
        for line in f:
            line=line.strip("\n")
            graph_indic[i]=int(line)
            i+=1

    filename_nodes=prefix + '_node_labels.txt'
    node_labels=[]
    try:
        with open(filename_nodes) as f:
            for line in f:
                line=line.strip("\n")
                node_labels+=[int(line) - 1]
        num_unique_node_labels = max(node_labels) + 1
    except IOError:
        print('No node labels')

    filename_node_attrs=prefix + '_node_attributes.txt'
    node_attrs=[]
    try:
        with open(filename_node_attrs) as f:
            for line in f:
                line = line.strip("\s\n")
                attrs = [float(attr) for attr in re.split("[,\s]+", line) if not attr == '']
                node_attrs.append(np.array(attrs))
    except IOError:
        print('No node attributes')

    label_has_zero = False
    filename_graphs=prefix + '_graph_labels.txt'
    graph_labels=[]

    # assume that all graph labels appear in the dataset
    #(set of labels don't have to be consecutive)
    label_vals = []
    with open(filename_graphs) as f:
        for line in f:
            line=line.strip("\n")
            val = int(line)
            #if val == 0:
            #    label_has_zero = True
            if val not in label_vals:
                label_vals.append(val)
            graph_labels.append(val)
    #graph_labels = np.array(graph_labels)
    label_map_to_int = {val: i for i, val in enumerate(label_vals)}
    graph_labels = np.array([label_map_to_int[l] for l in graph_labels])
    #if label_has_zero:
    #    graph_labels += 1

    filename_adj=prefix + '_A.txt'
    adj_list={i:[] for i in range(1,len(graph_labels)+1)}
    index_graph={i:[] for i in range(1,len(graph_labels)+1)}
    num_edges = 0
    with open(filename_adj) as f:
        for line in f:
            line=line.strip("\n").split(",")
            e0,e1=(int(line[0].strip(" ")),int(line[1].strip(" ")))
            adj_list[graph_indic[e0]].append((e0,e1))
            index_graph[graph_indic[e0]]+=[e0,e1]
            num_edges += 1
    for k in index_graph.keys():
        index_graph[k]=[u-1 for u in set(index_graph[k])]

    graphs=[]
    for i in range(1,1+len(adj_list)):
        # indexed from 1 here
        G=nx.from_edgelist(adj_list[i])
        if max_nodes is not None and G.number_of_nodes() > max_nodes:
            continue

        # add features and labels
        G.graph['label'] = graph_labels[i-1]
        for u in G.nodes():
            if len(node_labels) > 0:
                node_label_one_hot = [0] * num_unique_node_labels
                node_label = node_labels[u-1]
                node_label_one_hot[node_label] = 1
                G.node[u]['label'] = node_label_one_hot
            if len(node_attrs) > 0:
                G.node[u]['feat'] = node_attrs[u-1]
        if len(node_attrs) > 0:
            G.graph['feat_dim'] = node_attrs[0].shape[0]

        # relabeling
        mapping={}
        it=0
        if float(nx.__version__)<2.0:
            for n in G.nodes():
                mapping[n]=it
                it+=1
        else:
            for n in G.nodes:
                mapping[n]=it
                it+=1

        # indexed from 0
        graphs.append(nx.relabel_nodes(G, mapping))
    return graphs


def create_graph_from_points(points, index):
    xs = iter(points[index,:,0,0])
    ys = iter(points[index,:,0,1])
    zs = iter(points[index,:,0,2])
    G = nx.grid_graph(dim=[xs, ys, zs])
    G.graph['label'] = 1

    # relabeling
    mapping={}
    it=0
    if float(nx.__version__)<2.0:
        for n in G.nodes():
            mapping[n]=it
            it+=1
    else:
        for n in G.nodes:
            mapping[n]=it
            it+=1

    # indexed from 0
    return nx.relabel_nodes(G, mapping)

def read_mesh_file(datadir, dataname):
    '''
    Reads the train and validation file at data/mesh/Xtrn.npz
    Returns:
        List of networkx objects with graph and node labels
    '''
    train_file = os.path.join(datadir, dataname, 'Xtrn.npz')
    train_np_file = np.load(train_file)
    train_points  = train_np_file[train_np_file.files[0]]
    graphs = []
    for i in range(train_points.shape[0])[:50]:
        graphs.append(create_graph_from_points(train_points, i))
    return graphs


def create_mesh_pickle(datadir, dataname):
    '''
    Reads the train and validation file at data/mesh/Xtrn.npz
    Saves the graph object for each datapoint in a pickle
    '''
    local_dir_path = os.path.join(datadir, dataname)
    pickle_dir = os.path.join(local_dir_path, "pickles", "train")
    os.system('mkdir -p ' + pickle_dir)
    train_file = os.path.join(local_dir_path, 'Xtrn.npz')
    train_np_file = np.load(train_file)
    train_points  = train_np_file[train_np_file.files[0]]
    graphs = []
    for i in range(train_points.shape[0]):
        pickle_file = open(os.path.join(pickle_dir, "train" + str(i)+"_graph.pkl"), 'wb')
        pickle.dump(create_graph_from_points(train_points, i), pickle_file, -1)
        pickle_file.close()
    # return graphs


def read_mesh_graph_pickle(datadir, dataname):
    local_dir_path = os.path.join(datadir, dataname)
    pickle_dir = os.path.join(local_dir_path, "pickles", "train")
    num_graphs = len(os.listdir(pickle_dir))
    graphs = []
    for i in range(num_graphs):
        fname = os.path.join(pickle_dir, "train" + str(i)+"_graph.pkl")
        f = open(fname, 'rb')
        graphs.append(pickle.load(f))
        f.close()
    return graphs


if __name__ == "__main__":
    create_mesh_pickle("data", "mesh")
    graphs = read_mesh_graph_pickle("data", "mesh")
    print(graphs)
