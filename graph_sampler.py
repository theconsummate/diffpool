import networkx as nx
import numpy as np
import torch
import torch.utils.data

class GraphSampler(torch.utils.data.Dataset):
    ''' Sample graphs and nodes in graph
    '''
    def __init__(self, G_list, features='default', normalize=True, assign_feat='default', max_num_nodes=0):
        self.G_list = G_list
        self.adj = []
        self.len = []
        # self.feature_all = []

        # self.assign_feat_all = []
        self.features_mode = features
        self.assign_feat_mode = assign_feat

        if max_num_nodes == 0:
            self.max_num_nodes = max([G.number_of_nodes() for G in self.G_list])
        else:
            self.max_num_nodes = max_num_nodes

        #if features == 'default':
        self.feat_dim = self.G_list[0].node[0]['feat'].shape[0]

        # adj matrix is common for every mesh
        self.adj = np.array(nx.to_numpy_matrix(self.G_list[0]))
        if normalize:
            sqrt_deg = np.diag(1.0 / np.sqrt(np.sum(self.adj, axis=0, dtype=float).squeeze()))
            self.adj = np.matmul(np.matmul(sqrt_deg, self.adj), sqrt_deg)
        self.len = self.G_list[0].number_of_nodes()

        # calcuate the first feature to obtain assing_feat_dim
        ff0, af0 = self.get_feat(0, self.features_mode, self.assign_feat_mode)
        self.assign_feat_dim = af0.shape[1]

        print("init complete")


        # self.feat_dim = self.feature_all[0].shape[1]

    def __len__(self):
        return len(self.G_list)


    '''lazy loading of features'''
    def get_feat(self, idx, features='default', assign_feat='default'):
        # feat matrix: max_num_nodes x feat_dim
        graph_feature = None
        graph_assign_feat = None
        G = self.G_list[idx]
        if features == 'default':
            f = np.zeros((self.max_num_nodes, self.feat_dim), dtype=float)
            for i,u in enumerate(G.nodes()):
                f[i,:] = G.node[u]['feat']
            graph_feature = f
        elif features == 'id':
            graph_feature = np.identity(self.max_num_nodes)
        elif features == 'deg-num':
            degs = np.sum(np.array(adj), 1)
            degs = np.expand_dims(np.pad(degs, [0, self.max_num_nodes - G.number_of_nodes()], 0),
                                  axis=1)
            graph_feature = degs
        elif features == 'deg':
            self.max_deg = 10
            degs = np.sum(np.array(adj), 1).astype(int)
            degs[degs>max_deg] = max_deg
            feat = np.zeros((len(degs), self.max_deg + 1))
            feat[np.arange(len(degs)), degs] = 1
            feat = np.pad(feat, ((0, self.max_num_nodes - G.number_of_nodes()), (0, 0)),
                    'constant', constant_values=0)

            f = np.zeros((self.max_num_nodes, self.feat_dim), dtype=float)
            for i,u in enumerate(G.nodes()):
                f[i,:] = G.node[u]['feat']

            feat = np.concatenate((feat, f), axis=1)

            graph_feature = feat
        elif features == 'struct':
            self.max_deg = 10
            degs = np.sum(np.array(adj), 1).astype(int)
            degs[degs>10] = 10
            feat = np.zeros((len(degs), self.max_deg + 1))
            feat[np.arange(len(degs)), degs] = 1
            degs = np.pad(feat, ((0, self.max_num_nodes - G.number_of_nodes()), (0, 0)),
                    'constant', constant_values=0)

            clusterings = np.array(list(nx.clustering(G).values()))
            clusterings = np.expand_dims(np.pad(clusterings,
                                                [0, self.max_num_nodes - G.number_of_nodes()],
                                                'constant'),
                                         axis=1)
            g_feat = np.hstack([degs, clusterings])
            if 'feat' in G.node[0]:
                node_feats = np.array([G.node[i]['feat'] for i in range(G.number_of_nodes())])
                node_feats = np.pad(node_feats, ((0, self.max_num_nodes - G.number_of_nodes()), (0, 0)),
                                    'constant')
                g_feat = np.hstack([g_feat, node_feats])

            graph_feature = g_feat

        if assign_feat == 'id':
            graph_assign_feat = np.hstack((np.identity(self.max_num_nodes), graph_feature))
        else:
            graph_assign_feat = graph_feature

        return graph_feature, graph_assign_feat

    def __getitem__(self, idx):
        adj = self.adj[idx]
        num_nodes = adj.shape[0]
        adj_padded = np.zeros((self.max_num_nodes, self.max_num_nodes))
        adj_padded[:num_nodes, :num_nodes] = adj

        feat_idx, assign_feat_idx = self.get_feat(idx, self.features_mode, self.assign_feat_mode)
        print("got item")

        # use all nodes for aggregation (baseline)

        return {'adj':adj_padded,
                'feats':feat_idx.copy(),
                'label':self.G_list[idx].graph['label'],
                'num_nodes': num_nodes,
                'assign_feats':assign_feat_idx.copy()}

