import h5py
from time import time
from tqdm import tqdm

### Numerical Packages
import numpy as np

### Graph Network Packages
import nmslib
import networkx as nx

### PyTorch / PyG
import torch

# 构图 + Curv
import os
from GraphRicciCurvature.OllivierRicci import OllivierRicci
label='1'
h5_path = ''
save_path=''
if not os.path.exists(save_path):
    os.makedirs(save_path)



class Hnsw:
    def __init__(self, space='cosinesimil', index_params=None,
                 query_params=None, print_progress=True):
        self.space = space
        self.index_params = index_params
        self.query_params = query_params
        self.print_progress = print_progress

    def fit(self, X):
        index_params = self.index_params
        if index_params is None:
            index_params = {'M': 16, 'post': 0, 'efConstruction': 400}

        query_params = self.query_params
        if query_params is None:
            query_params = {'ef': 90}

        # this is the actual nmslib part, hopefully the syntax should
        # be pretty readable, the documentation also has a more verbiage
        # introduction: https://nmslib.github.io/nmslib/quickstart.html
        index = nmslib.init(space=self.space, method='hnsw')
        index.addDataPointBatch(X)
        index.createIndex(index_params, print_progress=self.print_progress)
        index.setQueryTimeParams(query_params)

        self.index_ = index
        self.index_params_ = index_params
        self.query_params_ = query_params
        return self

    def query(self, vector, topn):
        # the knnQuery returns indices and corresponding distance
        # we will throw the distance away for now
        indices, dist = self.index_.knnQuery(vector, k=topn)
        return indices,dist


def pt2graph(wsi_h5, radius=9):
    from torch_geometric.data import Data as geomData
    from itertools import chain
    coords, features = np.array(wsi_h5['coords']), np.array(wsi_h5['features'])
    assert coords.shape[0] == features.shape[0]
    num_patches = coords.shape[0]

    t1=time()
    model = Hnsw(space='l2')
    model.fit(coords)
    a = np.repeat(range(num_patches), radius - 1)
    b = np.fromiter(chain(*[model.query(coords[v_idx], topn=radius)[0][1:] for v_idx in range(num_patches)]), dtype=int)
    edge_spatial = torch.Tensor(np.stack([a, b])).type(torch.LongTensor)
    t2=time()
    # print('Hnsw took {} seconds'.format(t2-t1))

    model = Hnsw(space='l2')
    model.fit(features)
    a = np.repeat(range(num_patches), radius - 1)
    b = np.fromiter(chain(*[model.query(coords[v_idx], topn=radius)[0][1:] for v_idx in range(num_patches)]), dtype=int)
    edge_latent = torch.Tensor(np.stack([a, b])).type(torch.LongTensor)

    G = nx.Graph()

    # 添加节点
    num_nodes = features.shape[0]
    G.add_nodes_from(range(num_nodes))

    # 添加边（空间边）（可替换成latent边，可加入权重）
    for edge in edge_spatial.T.tolist():
        node1, node2 = edge
        G.add_edge(node1, node2)
    for node, feature_value in enumerate(features):
        G.nodes[node]['feature'] = feature_value.tolist()

    t3=time()
    orc = OllivierRicci(G, alpha=0.5, verbose="INFO")
    orc.compute_ricci_curvature()

    edge_cuva = []
    for a in range(edge_spatial.shape[1]):
        if edge_spatial[0,a] == edge_spatial[1,a]:
            edge_cuva.append(1)
        else:
            row = edge_spatial[0,a].numpy()
            col = edge_spatial[1,a].numpy()
            edge_cuva.append(orc.G[int(row)][int(col)]["ricciCurvature"])
    # print(edge_cuva)
    t4=time()

    # print('Curvature took {} seconds'.format(t4-t3))

    G = geomData(x=torch.Tensor(features),
                 edge_index=edge_spatial,
                 edge_attr=None,
                 edge_latent=edge_latent,
                 curva=torch.tensor(edge_cuva, dtype=torch.float),
                 centroid=torch.Tensor(coords))
    return G,t2-t1,t4-t3

def createDir_h5toPyG(h5_path, save_path):
    pbar = tqdm(os.listdir(h5_path))
    T_construct_graph = 0
    T_curvature = 0
    for h5_fname in pbar:
        pbar.set_description('%s - Creating Graph' % (h5_fname[:-3]))
        try:
            # print(os.path.join(h5_path, h5_fname))
            if os.path.exists(os.path.join(save_path, h5_fname[:-3]+'.pt')):
                continue
            wsi_h5 = h5py.File(os.path.join(h5_path, h5_fname), "r")
            G,t1,t2 = pt2graph(wsi_h5)
            T_construct_graph +=t1
            T_curvature +=t2
            torch.save(G, os.path.join(save_path, h5_fname[:-3]+'.pt'))
            print('Saved graph to', os.path.join(save_path, h5_fname[:-3]+'.pt'))
            wsi_h5.close()
        except OSError:
            pbar.set_description('%s - Broken H5' % (h5_fname[:12]))
            print(h5_fname, 'Broken')
    print('Graph construction Time: %.2f seconds' % (T_construct_graph))
    print('Curvature Time: %.2f seconds' % (T_curvature))


createDir_h5toPyG(h5_path, save_path)

# %%



