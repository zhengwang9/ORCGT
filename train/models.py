from torch.nn import Linear, LayerNorm, ReLU
from torch_geometric.nn import GCNConv, GraphConv, GatedGraphConv, GATConv, SGConv, GINConv, GENConv, DeepGCNLayer
from torch_geometric.nn import DenseGCNConv, dense_diff_pool
from torch_geometric.nn import EdgePooling
from model_utils import *
from torch_scatter import scatter_add, scatter_mean
from torch_sparse import coalesce

from rewritten_GCN import GCNConv_curv

class NormalizeFeaturesV2(object):
    r"""Column-normalizes node features to sum-up to one."""

    def __call__(self, data):
        data.x[:, :12] = data.x[:, :12] / data.x[:, :12].max(0, keepdim=True)[0]
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

class NormalizeEdgesV2(object):
    r"""Column-normalizes node features to sum-up to one."""

    def __call__(self, data):
        data.edge_attr = data.edge_attr.type(torch.cuda.FloatTensor)
        data.edge_attr = data.edge_attr / data.edge_attr.max(0, keepdim=True)[0]
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

class PatchGCN(torch.nn.Module):
    def __init__(self, input_dim=768, num_layers=4, edge_agg='spatial', multires=False, resample=0,
        fusion=None, num_features=768, hidden_dim=256, linear_dim=64, use_edges=False, pool=False, dropout=0.25, n_classes=2):
        super(PatchGCN, self).__init__()
        self.use_edges = use_edges
        self.fusion = fusion
        self.pool = pool
        self.edge_agg = edge_agg
        self.multires = multires
        self.num_layers = num_layers-1
        self.resample = resample

        if self.resample > 0:
            self.fc = nn.Sequential(*[nn.Dropout(self.resample), nn.Linear(num_features, 256), nn.ReLU(), nn.Dropout(0.25)])
        else:
            self.fc = nn.Sequential(*[nn.Linear(num_features, 256), nn.ReLU(), nn.Dropout(0.25)])

        self.layers = torch.nn.ModuleList()
        for i in range(1, self.num_layers+1):
            conv = GENConv(hidden_dim, hidden_dim, aggr='softmax',
                           t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm = LayerNorm(hidden_dim, elementwise_affine=True)
            act = ReLU(inplace=True)
            layer = DeepGCNLayer(conv, norm, act, block='res', dropout=0.1, ckpt_grad=i % 3)
            self.layers.append(layer)



        self.path_phi = nn.Sequential(*[nn.Linear(hidden_dim*num_layers, hidden_dim*num_layers), nn.ReLU(), nn.Dropout(0.25)])

        self.path_attention_head = Attn_Net_Gated(L=hidden_dim*num_layers, D=hidden_dim*num_layers, dropout=dropout, n_classes=1)
        self.path_rho = nn.Sequential(*[nn.Linear(hidden_dim*num_layers, hidden_dim*num_layers), nn.ReLU(), nn.Dropout(dropout)])

        self.classifier = torch.nn.Linear(hidden_dim*num_layers, n_classes)    

    def forward(self,  x):
        data = x
        if self.edge_agg == 'spatial':
            edge_index = data.edge_index
        elif self.edge_agg == 'latent':
            edge_index = data.edge_latent

        batch = data.batch
        edge_attr = None
        #edge_attr=data.edge_attr
        x = self.fc(data.x)
        x_ = x 
        

        x = self.layers[0].conv(x_, edge_index, edge_attr)
        x_ = torch.cat([x_, x], axis=1)
        for layer in self.layers[1:]:
            x = layer(x, edge_index, edge_attr)
            x_ = torch.cat([x_, x], axis=1)
        
        h_path = x_
        h_path = self.path_phi(h_path)

        A_path, h_path = self.path_attention_head(h_path)
        A_path = torch.transpose(A_path, 1, 0)
        h_path = torch.mm(F.softmax(A_path, dim=1), h_path)
        h = self.path_rho(h_path).squeeze()
        logits  = self.classifier(h).unsqueeze(0) # logits needs to be a [1 x 4] vector 
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        return logits,Y_hat,F.softmax(A_path, dim=1)




def reset_parameters(w):
    stdv = 1. / math.sqrt(w.size(0))
    w.data.uniform_(-stdv, stdv) 

class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq, msk=None):
        if msk is None:
            return torch.mean(seq, 0)
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.sum(seq * msk, 0) / torch.sum(msk)

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MLP, self).__init__()

        self.linear1 = nn.Linear(in_channels, 2 * out_channels)
        self.linear2 = nn.Linear(2 * out_channels, out_channels)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)

        return x

def func_k(curve):
    k = [1, 2, 3, 4, 5, 6,7, 8, 9,10]

    for i in range(len(k)):
        cur = (1+torch.exp(-k[i]*curve))/2

        if i == 0:
            Multi_curve = cur.unsqueeze(0).T
        else:
            Multi_curve = torch.cat((Multi_curve, cur.unsqueeze(0).T), 1)
        #Multi_curve = curve.unsqueeze(0).T
    #print(Multi_curve.shape)
    return Multi_curve


class PatchGCN_curv(torch.nn.Module):
    def __init__(self, input_dim=768, num_layers=4, edge_agg='spatial', multires=False, resample=0,
        fusion=None, num_features=768, hidden_dim=256, linear_dim=64, use_edges=False, pool=False, dropout=0.25, n_classes=2):
        super(PatchGCN_curv, self).__init__()
        self.use_edges = use_edges
        self.fusion = fusion
        self.pool = pool
        self.edge_agg = edge_agg
        self.multires = multires
        self.num_layers = num_layers-1
        self.resample = resample

        if self.resample > 0:
            self.fc = nn.Sequential(*[nn.Dropout(self.resample), nn.Linear(num_features, 256), nn.ReLU(), nn.Dropout(0.25)])
        else:
            self.fc = nn.Sequential(*[nn.Linear(num_features, hidden_dim), nn.ReLU(), nn.Dropout(0.25)])

        self.layers = torch.nn.ModuleList()
        for i in range(1, self.num_layers+1):
            conv = GENConv(hidden_dim, hidden_dim, aggr='softmax',
                           t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm = LayerNorm(hidden_dim, elementwise_affine=True)
            act = ReLU(inplace=True)
            layer = DeepGCNLayer(conv, norm, act, block='res', dropout=0.1, ckpt_grad=i % 3)
            self.layers.append(layer)

        #num_layers = 3 # 单层的时候使用
        self.path_phi = nn.Sequential(*[nn.Linear(hidden_dim*num_layers, hidden_dim*num_layers), nn.ReLU(), nn.Dropout(0.25)])

        self.path_attention_head = Attn_Net_Gated(L=hidden_dim*num_layers, D=hidden_dim*num_layers, dropout=dropout, n_classes=1)
        self.path_rho = nn.Sequential(*[nn.Linear(hidden_dim*num_layers, hidden_dim*num_layers), nn.ReLU(), nn.Dropout(dropout)])

        self.classifier = torch.nn.Linear(hidden_dim*num_layers, n_classes)    

        self.encoder_o1 = GCNConv_curv(hidden_dim,hidden_dim)
        self.lin1 = nn.Linear(10, 1)

        self.encoder_o2 = GCNConv_curv(hidden_dim*2,hidden_dim)
        self.lin2 = nn.Linear(10, 1)

        self.encoder_o3 = GCNConv_curv(hidden_dim,hidden_dim)
        self.lin3 = nn.Linear(10, 1)

        self.dropout = 0.5

    # def forward(self,  x):
    #     data = x
    #     if self.edge_agg == 'spatial':
    #         edge_index = data.edge_index
    #     elif self.edge_agg == 'latent':
    #         edge_index = data.edge_latent
    #
    #     batch = data.batch
    #     edge_attr = None
    #     #edge_attr=data.edge_attr
    #
    #     #x = data.x
    #     x = self.fc(data.x) # 不要fc
    #     x_ = x
    #
    #     ######## curva
    #     x_o, adj, curvature = x_, edge_index,data.curva
    #     curva1 = self.lin1(func_k(curvature))
    #     curva1 = F.dropout(curva1, self.dropout, training=self.training)
    #     x1_o = F.relu(self.encoder_o1(x_o, adj, curva1))
    #     x1_o = F.dropout(x1_o, self.dropout, training=self.training)
    #     x_ = torch.cat([x_, x1_o], axis=1)
    #
    #     adj, curvature = edge_index,data.curva
    #     curva2 = self.lin2(func_k(curvature))
    #     curva2 = F.dropout(curva2, self.dropout, training=self.training)
    #     x2_o = F.relu(self.encoder_o2(x_, adj, curva2))
    #     x2_o = F.dropout(x2_o, self.dropout, training=self.training)
    #     x_ = torch.cat([x_, x2_o], axis=1)
    #
    #     ####### curva + conv
    #     # x = self.layers[0].conv(x_, edge_index, edge_attr)
    #     # #x_ = torch.cat([x_, x], axis=1)
    #     # layer = self.layers[1]
    #     # x = layer(x, edge_index, edge_attr)
    #     # x_ = torch.cat([x_, x], axis=1)
    #
    #     # adj, curvature = edge_index,data.curva
    #     # curva2 = self.lin2(func_k(curvature))
    #     # curva2 = F.dropout(curva2, self.dropout, training=self.training)
    #     # x2_o = F.relu(self.encoder_o2(x_, adj, curva2))
    #     # x2_o = F.dropout(x2_o, self.dropout, training=self.training)
    #     # x_ = torch.cat([x_, x2_o], axis=1)
    #
    #     ###### conv
    #     # x = self.layers[0].conv(x_, edge_index, edge_attr)
    #     # x_ = torch.cat([x_, x], axis=1)
    #     # for layer in self.layers[1:]:
    #     #     x = layer(x, edge_index, edge_attr)
    #     #     x_ = torch.cat([x_, x], axis=1)
    #
    #     h_path = x_
    #     h_path = self.path_phi(h_path)
    #
    #     A_path, h_path = self.path_attention_head(h_path)
    #     A_path = torch.transpose(A_path, 1, 0)
    #     h_path = torch.mm(F.softmax(A_path, dim=1), h_path)
    #     h = self.path_rho(h_path).squeeze()
    #     logits  = self.classifier(h).unsqueeze(0) # logits needs to be a [1 x 4] vector
    #     Y_hat = torch.topk(logits, 1, dim = 1)[1]
    #     return logits,Y_hat,A_path

    def forward_zzf(self,  x):
        data = x
                
        if self.edge_agg == 'spatial':
            edge_index = data.edge_index
        elif self.edge_agg == 'latent':
            edge_index = data.edge_latent

        batch = data.batch
        edge_attr = None

        #x = data.x
        x = self.fc(data.x)
        x_ = x 

        ###### curva
        x_o, adj, curvature = x_, edge_index,data.curva
        curva1 = self.lin1(func_k(curvature))
        curva1 = F.dropout(curva1, self.dropout, training=self.training)
        x1_o = F.relu(self.encoder_o1(x_o, adj, curva1))
        x1_o = F.dropout(x1_o, self.dropout, training=self.training)
        x_ = torch.cat([x_, x1_o], axis=1)
        #print(x_.shape)

        curva2 = self.lin2(func_k(curvature))
        curva2 = F.dropout(curva2, self.dropout, training=self.training)
        x2_o = F.relu(self.encoder_o2(x_, adj, curva2))
        x2_o = F.dropout(x2_o, self.dropout, training=self.training)
        x_ = torch.cat([x_, x2_o], axis=1)

        ####### curva + conv
        # x = self.layers[0].conv(x_, edge_index, edge_attr)
        # #x_ = torch.cat([x_, x], axis=1)
        # layer = self.layers[1]
        # x = layer(x, edge_index, edge_attr)
        # x_ = torch.cat([x_, x], axis=1)

        # adj, curvature = edge_index,data.curva
        # curva2 = self.lin2(func_k(curvature))
        # curva2 = F.dropout(curva2, self.dropout, training=self.training)
        # x2_o = F.relu(self.encoder_o2(x_, adj, curva2))
        # x2_o = F.dropout(x2_o, self.dropout, training=self.training)
        # x_ = torch.cat([x_, x2_o], axis=1)

        ###### conv
        # x = self.layers[0].conv(x_, edge_index, edge_attr)
        # x_ = torch.cat([x_, x], axis=1)
        # for layer in self.layers[1:]:
        #     x = layer(x, edge_index, edge_attr)
        #     x_ = torch.cat([x_, x], axis=1)
        
        h_path = x_
        h_path = self.path_phi(h_path)

        A_path, h_path = self.path_attention_head(h_path)
        A_path = torch.transpose(A_path, 1, 0)
        h_path = torch.mm(F.softmax(A_path, dim=1), h_path)
        h = self.path_rho(h_path).squeeze()
        logits  = self.classifier(h).unsqueeze(0) # logits needs to be a [1 x 4] vector 
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        return logits,Y_hat, h_path,A_path



def build_degree_matrix(adjacency_matrix):
    # 计算每个节点的度
    degree = torch.sum(adjacency_matrix, dim=1)

    # 构建度矩阵
    degree_matrix = torch.diag(degree)

    return degree_matrix.to(device[0])


def build_symmetric_normalized_laplacian(adjacency_matrix):
    # 构建度矩阵
    degree_matrix = build_degree_matrix(adjacency_matrix)

    # 计算度矩阵的平方根的逆矩阵
    sqrt_inverse_degree_matrix = torch.diag(torch.sqrt(1.0 / degree_matrix.diagonal())).to(device[0])

    # 计算标准化的对称拉普拉斯矩阵
    laplacian_matrix = torch.eye(adjacency_matrix.shape[0]).to(device[0]) - torch.matmul(
        torch.matmul(sqrt_inverse_degree_matrix, adjacency_matrix), sqrt_inverse_degree_matrix)

    return laplacian_matrix.to(device[0])



class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 n_layers, normalize=False, lin=True):
        super(GNN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(DenseGCNConv(in_channels, hidden_channels, normalize))
        for i in range(n_layers - 1):
            self.convs.append(DenseGCNConv(hidden_channels, hidden_channels, normalize))
        self.convs.append(DenseGCNConv(hidden_channels, out_channels, normalize))

        self.bns = torch.nn.ModuleList()
        for i in range(n_layers):
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.bns.append(torch.nn.BatchNorm1d(out_channels))

    def forward(self, x, adj, mask=None):
        x = x.unsqueeze(0)
        batch_size, num_nodes, in_channels = x.size()

        for layer_idx in range(len(self.convs)):
            x = F.relu(self.convs[layer_idx](x, adj, mask))

        return x




class ORCG(torch.nn.Module):
    '''

    '''
    def __init__(self, input_dim=768, num_layers=4, edge_agg='spatial', multires=False, resample=0,
                 fusion=None, num_features=768, hidden_dim=256, linear_dim=64, use_edges=False, pool=False,
                 dropout=0.25, n_classes=2):
        super(ORCG, self).__init__()
        self.use_edges = use_edges
        self.fusion = fusion
        self.pool = pool
        self.edge_agg = edge_agg
        self.multires = multires
        self.num_layers = num_layers - 1
        self.resample = resample

        if self.resample > 0:
            self.fc = nn.Sequential(
                *[nn.Dropout(self.resample), nn.Linear(num_features, 256), nn.ReLU(), nn.Dropout(0.25)])
        else:
            self.fc = nn.Sequential(*[nn.Linear(num_features, hidden_dim), nn.ReLU(), nn.Dropout(0.25)])

        self.layers = torch.nn.ModuleList()
        for i in range(1, self.num_layers + 1):
            conv = GENConv(hidden_dim, hidden_dim, aggr='softmax',
                           t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm = LayerNorm(hidden_dim, elementwise_affine=True)
            act = ReLU(inplace=True)
            layer = DeepGCNLayer(conv,norm, act, block='res', dropout=0.1, ckpt_grad=i % 3)
            self.layers.append(layer)

        self.path_phi = nn.Sequential(
            *[nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.25)])

        self.path_attention_head = Attn_Net_Gated(L=hidden_dim, D=hidden_dim, dropout=dropout,
                                                  n_classes=1)
        self.path_rho = nn.Sequential(
            *[nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])

        self.classifier = torch.nn.Linear(hidden_dim, n_classes)
        self.edge_pool0 = EdgePooling(hidden_dim)
        self.edge_pool1 = EdgePooling(hidden_dim)
        self.edge_pool2 = EdgePooling(hidden_dim)
        self.encoder_o1 = GCNConv_curv(hidden_dim,hidden_dim)
        self.lin1 = nn.Linear(10, 1)
        # self.dropout = 0.5


        # self.pooling1 = DiffPool(1024, 1024)
        # self.pooling2 = DiffPool(1024, 512)

    def forward(self, x):
        data = x
        if self.edge_agg == 'spatial':
            edge_index = data.edge_index
        elif self.edge_agg == 'latent':
            edge_index = data.edge_latent

        #构建邻接矩阵
        # adj=torch.zeros((data.num_nodes, data.num_nodes)).to(device[0])
        # for i in range(data.num_nodes):
        #     adj[i,0:8]=data.edge_index[1,i*8:i*8+8]
        #     adj[i*8:i*8+8,0:8]=data.edge_index[1,i*8:i*8+8]
        # laplacian=build_symmetric_normalized_laplacian(adj_matrix)

        batch = data.batch
        edge_attr = None
        # edge_attr=data.edge_attr
        x = self.fc(data.x)

        ###### curva
        x_o, adj, curvature = x, edge_index, data.curva
        curva1 = self.lin1(func_k(curvature))
        x1_o = F.relu(self.encoder_o1(x_o, adj, curva1))
        x=x1_o
        pool_id = 1

        x,edge_index,batch,_=self.edge_pool0(x, edge_index, batch)

        for layer in self.layers[1:]:
            x = layer.conv(x, edge_index, edge_attr)
            x, edge_index, batch, _ = getattr(self, 'edge_pool'+str(pool_id))(x, edge_index, batch)
            # x=layer.norm(x)
            # x=layer.act(x)
            pool_id += 1

        # h_path = x_
        h_path = x
        h_path = self.path_phi(h_path)

        A_path, h_path = self.path_attention_head(h_path)
        A_path = torch.transpose(A_path, 1, 0)
        h_path = torch.mm(F.softmax(A_path, dim=1), h_path)
        h = self.path_rho(h_path).squeeze()
        logits = self.classifier(h).unsqueeze(0)  # logits needs to be a [1 x 4] vector
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        return logits, Y_hat, h_path

    def forward_vis(self, x):
        data = x
        if self.edge_agg == 'spatial':
            edge_index = data.edge_index
        elif self.edge_agg == 'latent':
            edge_index = data.edge_latent
        unpool_infos=[]
        #构建邻接矩阵
        batch = data.batch
        edge_attr = None
        # edge_attr=data.edge_attr
        x = self.fc(data.x)

        ###### curva
        x_o, adj, curvature = x, edge_index, data.curva
        curva1 = self.lin1(func_k(curvature))
        x1_o = F.relu(self.encoder_o1(x_o, adj, curva1))
        x=x1_o
        pool_id = 1

        x,edge_index,batch,unpool_info=self.edge_pool0(x, edge_index, batch)
        unpool_infos.append(unpool_info)

        for layer in self.layers[1:]:
            x = layer.conv(x, edge_index, edge_attr)
            x, edge_index, batch, unpool_info = getattr(self, 'edge_pool'+str(pool_id))(x, edge_index, batch)
            unpool_infos.append(unpool_info)
            # x=layer.norm(x)
            # x=layer.act(x)
            pool_id += 1

        # h_path = x_
        h_path = x
        h_path = self.path_phi(h_path)

        A_path, h_path = self.path_attention_head(h_path)
        A_path = torch.transpose(A_path, 1, 0)
        h_path = torch.mm(F.softmax(A_path, dim=1), h_path)

        attn=F.softmax(A_path, dim=1)
        node_cluster = unpool_infos[0].cluster
        #
        for i in range(1, len(unpool_infos)):
            node_cluster = unpool_infos[i].cluster[node_cluster.long()]
        node_attscores = attn[node_cluster.long()]

        h = self.path_rho(h_path).squeeze()
        logits = self.classifier(h).unsqueeze(0)  # logits needs to be a [1 x 4] vector
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        return logits, Y_hat, h_path,node_attscores





class MyEdgePooling(EdgePooling):
    def __init__(self, in_channels, edge_score_method=None, dropout=0,
                 add_to_edge_score=0.5):
        super().__init__(in_channels, edge_score_method=None, dropout=0, add_to_edge_score=0.5)
        self.lin = torch.nn.Linear(in_channels, in_channels)
        self.relu = torch.nn.ReLU()

    def forward(self, x, edge_index, batch):
        a,b = x[edge_index[0]], x[edge_index[1]]
        e = (a*b).sum(dim=1)
        # e = self.lin(e).view(-1)
        e = F.dropout(e, p=self.dropout, training=self.training)
        e = self.compute_edge_score(e, edge_index, x.size(0))
        e = e + self.add_to_edge_score

        x, edge_index, batch, unpool_info = self.__merge_edges__(x, edge_index, batch, e)

        return x, edge_index, batch, unpool_info

    def __merge_edges__(self, x, edge_index, batch, edge_score):
        nodes_remaining = set(range(x.size(0)))

        cluster = torch.empty_like(batch, device=torch.device('cpu'))
        edge_argsort = torch.argsort(edge_score, descending=True)

        # Iterate through all edges, selecting it if it is not incident to
        # another already chosen edge.
        i = 0
        new_edge_indices = []
        edge_index_cpu = edge_index.cpu()
        for edge_idx in edge_argsort.tolist():
            source = edge_index_cpu[0, edge_idx].item()
            if source not in nodes_remaining:
                continue

            target = edge_index_cpu[1, edge_idx].item()
            if target not in nodes_remaining:
                continue

            new_edge_indices.append(edge_idx)

            cluster[source] = i
            nodes_remaining.remove(source)

            if source != target:
                cluster[target] = i
                nodes_remaining.remove(target)

            i += 1

        # The remaining nodes are simply kept.
        for node_idx in nodes_remaining:
            cluster[node_idx] = i
            i += 1
        cluster = cluster.to(x.device)
        # We compute the new features as an addition of the old ones.
        new_x = scatter_mean(x, cluster, dim=0, dim_size=i)
        new_x = self.relu(self.lin(new_x))
        new_edge_score = edge_score[new_edge_indices]
        if len(nodes_remaining) > 0:
            remaining_score = x.new_ones(
                (new_x.size(0) - len(new_edge_indices),))
            new_edge_score = torch.cat([new_edge_score, remaining_score])
        new_x = new_x * new_edge_score.view(-1, 1)

        N = new_x.size(0)
        new_edge_index, _ = coalesce(cluster[edge_index], None, N, N)

        new_batch = x.new_empty(new_x.size(0), dtype=torch.long)
        new_batch = new_batch.scatter_(0, cluster, batch)

        unpool_info = self.unpool_description(edge_index=edge_index,
                                              cluster=cluster, batch=batch,
                                              new_edge_score=new_edge_score)

        return new_x, new_edge_index, new_batch, unpool_info
        # We compute the new featu


def func_k_1(curve):
    k = [1, 2, 3, 4, 5]

    for i in range(len(k)):
        cur = (1 + torch.exp(-k[i] * curve)) / 2

        if i == 0:
            Multi_curve = cur.unsqueeze(0).T
        else:
            Multi_curve = torch.cat((Multi_curve, cur.unsqueeze(0).T), 1)

    cur = -curve
    Multi_curve = torch.cat((Multi_curve, cur.unsqueeze(0).T), 1)

    cur = torch.pow(curve, 2) + 1
    Multi_curve = torch.cat((Multi_curve, cur.unsqueeze(0).T), 1)

    return Multi_curve




