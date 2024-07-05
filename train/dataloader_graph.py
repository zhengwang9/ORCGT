# 将数据组织成dataset形式，将多个patch合并成一个tensor
# %%
import numpy as np
import torch
import torch.utils.data as data_utils
import torch_geometric.utils as utils
import h5py
import os
import csv
import pandas as pd

import torch
from torch_geometric.data import Dataset, download_url
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch_scatter import scatter, segment_csr, gather_csr
import warnings

warnings.filterwarnings("ignore", category=Warning)
from torch_geometric.datasets import TUDataset
 # %%


def extract_node_feature(data, reduce='add'):
    if reduce in ['mean', 'max', 'add']:
        data.x = scatter(data.edge_attr,
                         data.edge_index[0],
                         dim=0,
                         dim_size=data.num_nodes,
                         reduce=reduce)
    else:
        raise Exception('Unknown Aggregation Type')
    return data


class StasDataset_gcn(Dataset):
    def __init__(self, root,labels_path, reduce='add',transform=None, pre_transform=None):
        self.labels_path = labels_path
        super(StasDataset_gcn, self).__init__(root,transform,pre_transform)
        self.root = root
        # self.degree = degree
        # self.reduce =reduce
        # self.datalist=self.process()
    # 返回数据集源文件名，告诉原始的数据集存放在哪个文件夹下面，如果数据集已经存放进去了，那么就会直接从raw文件夹中读取。
    @property
    def raw_file_names(self):
        # pass # 不能使用pass，会报join() argument must be str or bytes, not 'NoneType'错误
        return []

    # 首先寻找processed_paths[0]路径下的文件名也就是之前process方法保存的文件名
    @property
    def processed_file_names(self):
        return []

    # 用于从网上下载数据集，下载原始数据到指定的文件夹下，自己的数据集可以跳过
    # def download(self):
    #     pass

    # 生成数据集所用的方法，程序第一次运行才执行并生成processed文件夹的处理过后数据的文件，否则必须删除已经生成的processed文件夹中的所有文件才会重新执行此函数
    # def process(self):
    #     pass
    def len(self):
        return len(os.listdir(self.root))

    def get(self,idx):
        files = os.listdir(self.root)
        # datalist = []
        for i,f in enumerate(files):
            if i!=idx:
                continue
            file_path = os.path.join(self.root, f)
            graph = torch.load(file_path)

            file_name, _ = os.path.splitext(f)
            label_dicts = csv.DictReader(open(self.labels_path, 'r'))
            for dict in label_dicts:
                if dict['file_name'] == file_name:
                    label = int(dict['STAS'])
                    Y = torch.tensor([label], dtype=torch.long)
                    # Y = torch.tensor([[label]], dtype=torch.long)
                    break
            # graph = Data(x=graph.x, edge_index=graph.edge_index, edge_latent=graph.edge_latent, centroid=graph.centroid)
            # if self.degree:
            #     degree=self.compute_degree(graph)
            graph = Data(x=graph.x, edge_index=graph.edge_index,edge_latent=graph.edge_latent, edge_attr=graph.edge_attr,y=Y,centroid=graph.centroid)
            # graph=extract_node_feature(graph,self.reduce)
            return graph
    def compute_degree(self,g):
        deg = 1. / torch.sqrt(1. + utils.degree(g.edge_index[0], g.num_nodes))
        return deg


class StasDataset_cv(Dataset):
    def __init__(self, root, cv_fold_id, labels_path, fold="fold_0", data_type='train', transform=None,
                 pre_transform=None):
        super(StasDataset_cv, self).__init__(root, transform, pre_transform)
        self.labels_path = labels_path
        self.root = root
        # root  = '/data3/cm/STAS/data/patch_level1/Split_graph_SimCLR_huandai_concat_level_12'
        self.fold = fold
        self.cv_fold_id = cv_fold_id
        # print(cv_fold_id)
        # print(fold)
        # print(data_type)
        self.data_type = data_type
        self.ids = self.cv_fold_id[self.fold][self.data_type]
        self.label_dicts = pd.read_csv(self.labels_path)

    @property
    def raw_file_names(self):
        # pass # 不能使用pass，会报join() argument must be str or bytes, not 'NoneType'错误
        return []

    # 首先寻找processed_paths[0]路径下的文件名也就是之前process方法保存的文件名
    @property
    def processed_file_names(self):
        return []

    def len(self):
        return len(self.ids)

    def get(self, idx):
        id = self.ids[idx]
        root_list = ['STAS0','STAS1']
        root_list = [os.path.join(self.root, i) for i in root_list]
        for i in root_list:
            if os.path.exists(os.path.join(i, id + '.ibl.pt')):
                graph = torch.load(os.path.join(i, id + '.ibl.pt'))

                label = int(self.label_dicts[self.label_dicts['file_name'] == id + '.ibl']['STAS'].values)
                Y = torch.tensor([label], dtype=torch.long)
                # graph = Data(x=graph.x, edge_index=graph.edge_index, edge_latent=graph.edge_latent,edge_attr=graph.edge_attr,
                #              centroid=graph.centroid,
                #              y=Y)
                graph = Data(x=graph.x, edge_index=graph.edge_index, edge_latent=graph.edge_latent,
                             centroid=graph.centroid,
                             y=Y)
                break
        return graph





class StasDataset_cv_curva(Dataset):
    def __init__(self, root, cv_fold_id, labels_path, fold="fold_0", data_type='train', transform=None,
                 pre_transform=None):
        super(StasDataset_cv_curva, self).__init__(root, transform, pre_transform)
        self.labels_path = labels_path
        self.root = root
        self.fold = fold
        self.cv_fold_id = cv_fold_id

        self.data_type = data_type
        self.ids = self.cv_fold_id[self.fold][self.data_type]
        self.label_dicts = pd.read_csv(self.labels_path)

    @property
    def raw_file_names(self):
        # pass # 不能使用pass，会报join() argument must be str or bytes, not 'NoneType'错误
        return []

    # 首先寻找processed_paths[0]路径下的文件名也就是之前process方法保存的文件名
    @property
    def processed_file_names(self):
        return []

    def len(self):
        return len(self.ids)

    def get(self, idx):
        id = self.ids[idx]
        root_list = ['STAS0','STAS1']
        root_list = [os.path.join(self.root, i) for i in root_list]
        graph_=''
        for i in root_list:
            if os.path.exists(os.path.join(i, id + '.pt')):
                graph = torch.load(os.path.join(i, id + '.pt'))

                label = int(self.label_dicts[self.label_dicts['file_name'] == id]['STAS'].values)
                Y = torch.tensor([label], dtype=torch.long)

                graph_ = Data(x=graph.x, edge_index=graph.edge_index, edge_latent=graph.edge_latent,
                             centroid=graph.centroid,curva = graph.curva,
                             y=Y)
                break
        return graph_,id


