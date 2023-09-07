from torch_geometric.data import Data
from random import choice
import torch 
import numpy as np
from torch_geometric.loader import DataLoader
import pandas as pd
from torch_geometric.utils import to_dense_adj,add_remaining_self_loops
from ot.utils import unif,dist
from ot.gromov import fused_gromov_wasserstein2
from typing import Union
from pickle import load

#### ------------------------------------------ Base ------------------------------------------####

class Instance():

    '''
    For generating instance of the dataset.
    Need a 'to_torch_data' method 
    '''

    def __init__(self):
        self.label = choice([0,1])

    def to_torch_data(self):
        if self.label == 0:
            x = torch.ones(2,2)
            edge_index= torch.tensor([[0, 1, 0, 1],[1, 0, 0, 1]], dtype=torch.long)
            return Data(x=x,edge_index=edge_index)
        elif self.label == 1:
            x = -torch.ones(2,2)
            edge_index= torch.tensor([[0, 1, 0, 1],[1, 0, 0, 1]], dtype=torch.long)
            return Data(x=x,edge_index=edge_index)


class Feature():
    '''
    A feature, for coloring TSNE or eval regression in the embedding
    '''

    def __init__(self):
        self.name = 'a feature'
        self.type = 'discrete' # ='continuous'
        self.dim = 2

    def __call__(self,x: Union[Instance,Data]):
        if x.label == 1:
            return torch.tensor([0.,1.])
        elif x.label == 0:
            return torch.tensor([1.,0.])


class Custom_Metric():
    '''
    A custom metric, for metric learning
    '''

    def __init__(self):
        self.name = 'A custom metric'

    def __call__(self,x: Union[Instance,Data],y: Union[Instance,Data]):
        if x.label == y.label:
            return 0.
        else:
            return 1.
        
class fgw_Metric():
    '''
    fgw metric, for metric learning
    '''

    def __init__(self, alpha):
        self.name = f'fgw : {alpha}'
        self.alpha = alpha

    def __call__(self,x: Union[Instance,Data],y:Union[Instance,Data]):

        if isinstance(x,Instance):
            g1 = x.to_torch_data()
        else:
            g1 = x
        if isinstance(y,Instance):
            g2 = y.to_torch_data()
        else:
            g2 = y

        M = dist(g1.x,g2.x)
        p1 = torch.tensor(unif(len(g1.x)))
        p2 = torch.tensor(unif(len(g2.x)))
        C1 = to_dense_adj(edge_index = g1.edge_index,max_num_nodes =len(g1.x)).squeeze()
        C2 = to_dense_adj(edge_index = g2.edge_index,max_num_nodes =len(g2.x)).squeeze()

        fgw = fused_gromov_wasserstein2(M,C1=C1,C2=C2,p=p1,q=p2,alpha=self.alpha).item()

        return fgw
  

class Dataset():
    
    def __init__(self, 
                 normalize_w: bool = False, # Normalize feature such that mean(GW) = mean(W)
                 custom_metric: bool = False, # Use a custom metric
                 alpha = Union[str,int], # Else alpha (int or 'auto')
                 load_distances = False): # Load precomputed distances (default = False for speed)

        '''
        To define a dataset, you need to overwrite the following components
        '''

        # --------------- Information about dataset --------------- #

        self.load_info()

        # --------------- Preprocessing --------------- #

        self.transform = self.get_transform(normalize_w)

        # --------------- Load precomputed --------------- #

        if load_distances:
            self.precomputed_distances = self.load_precomputed_distances(custom_metric,alpha)

        # --------------- Define Metrics --------------- #

        self.define_metric(custom_metric,alpha)
        if self.feature.type=="discrete":
            self.output_dim = self.feature.n_classes
        if self.feature.type=="continuous":
            self.output_dim = self.feature.dim

    def load_info(self):

        '''
        The following are the minimal information that define a dataset:
        - node_dim (int)
        - name (str)
        - size_qt005,size_qt095,size_qt050 (quantile of sizes of the graphs in the dataset)
        - default_alpha (float) default value of alpha
        '''

        self.node_dim = None

        self.name = 'a dataset'

        self.size_qt005 = None
        self.size_qt095 = None
        self.size_qt050 = None

        self.default_alpha = None # Default alpha for normalized_w = True and alpha = 'auto'
        self.normalization = None # Associated normalization i.e. avg distance

    def get_transform(self, normalize_w: bool):

        if normalize_w: # Normalization for mean(gw) = mean(fGW)
            unnormalized_gw = pd.read_csv('saves/'+self.name+'_Data/unnormalized_gw.csv')
            sigma = np.sqrt(unnormalized_gw['w'].mean()/unnormalized_gw['gw'].mean())
        else:
            sigma=1

        def transform(data: Data):
            data.x = data.x/sigma
            data.edge_index = add_remaining_self_loops(data.edge_index)[0]
            return data
        
        return transform

    def load_precomputed_distances(self,custom_metric,alpha):

        if not(custom_metric) and alpha == 'auto':
            # Load default value
            precomputed_distances = pd.read_csv('saves/'+self.name+'_Data/default_fgw.csv')
            #self.normalization = precomputed_distances['dist'].mean()
        else:
            # TO DO
            assert False

        return precomputed_distances

    def define_metric(self,custom_metric,alpha):
        self.metric = fgw_Metric(0.5)
        self.feature = Feature()


#### ------------------------------------------ Synthetic Dataset ------------------------------------------####


class SyntheticDataset(Dataset):
    
    def __init__(self, 
                 normalize_w: bool = False, # Normalize feature such that mean(GW) = mean(W)
                 custom_metric: bool = False, # Use a custom metric
                 alpha = Union[str,int], # Else alpha (int or 'auto')
                 load_distances = False): # Load precomputed distances (default = False for speed)

        super().__init__(normalize_w = normalize_w,
                              custom_metric = custom_metric,
                              alpha = alpha,
                              load_distances = load_distances)
        
        if custom_metric:
            self.name += '_custom'
        
        if load_distances:
            self.precomputed_instances = self.load_precomputed_instances()
            
    def load_precomputed_instances(self):
        with open('saves/'+self.name+f'_Data/precomputed_instances.pickle', 'rb') as handle:
            precomputed_instances = load(handle)
        return [self.transform(G) for G in precomputed_instances]
    
    def get_instance(self):
        return Instance()

    def get_dataloader(self,batchsize = 512, dataset_size = 100000):
        return DataLoader([self.transform(self.get_instance().to_torch_data()) for _ in range(dataset_size)],batch_size = batchsize,shuffle = True)
    
    def get_dataloader_ml(self,batchsize = 512):
        return DataLoader([(self.precomputed_instances[int(row['i'])],
                            self.precomputed_instances[int(row['j'])],
                            row['dist'])
                            for index, row in self.precomputed_distances.iterrows()],batch_size = batchsize,shuffle = True)

    def get_dataloader_reg(self,batchsize = 512, TSNE = False, dataset_size = 100000):
        all_instances = [self.get_instance() for _ in range(dataset_size)]
        return DataLoader([(self.transform(I.to_torch_data()),self.feature(I)) for I in all_instances],batch_size = batchsize)


 
#### ------------------------------------------ Real Dataset ------------------------------------------####

class RealDataset(Dataset):

    def __init__(self, 
                 normalize_w: bool = False, # Normalize feature such that mean(GW) = mean(W)
                 custom_metric: bool = False, # Use a custom metric
                 alpha = Union[str,int], # Else alpha (int or 'auto')
                 load_distances = False): # Load precomputed distances (default = False for speed)

        super().__init__(normalize_w = normalize_w,
                              custom_metric = custom_metric,
                              alpha = alpha,
                              load_distances = load_distances)
        
        self.data = self.load_dataset()

    
    def load_dataset(self):
        dataset = []
        return dataset   
    
    def get_dataloader(self,batchsize = 512, dataset_size = 100000):
        return DataLoader([G for G in self.data],batch_size = batchsize,shuffle = True)

    
    def get_dataloader_ml(self,batchsize = 512,train = True):
        return DataLoader([(self.data[int(row['i'])],
                            self.data[int(row['j'])],
                            row['dist'])
                            for index, row in self.precomputed_distances.iterrows()],batch_size = batchsize,shuffle = True)
    
    def get_dataloader_reg(self,batchsize = 512, TSNE = False, dataset_size = 100000):
        return DataLoader([(G,self.feature(G)) for G in self.data],batch_size = batchsize,shuffle = True)
