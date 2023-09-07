
#For some extremely weird reason importing in that order causes a bug on my machine
#from .base import RealDataset,Feature,fgw_Metric
#from ogb.graphproppred import PygGraphPropPredDataset

from torch_geometric.utils import to_dense_adj,add_remaining_self_loops

from ogb.graphproppred import PygGraphPropPredDataset
from datasets.base import RealDataset,Feature,fgw_Metric
from torch_geometric.loader import DataLoader

from ogb.graphproppred.mol_encoder import AtomEncoder

from torch_geometric.data import Data
import torch 
import numpy as np
import pandas as pd



class ogb_label(Feature):

    def __init__(self):
        self.name = 'ogb label'
        self.type = 'discrete' # ='continuous'
        self.scoring = 'roc_auc'
        self.n_classes = 2

    def __call__(self,G : Data):
        return torch.tensor(G.y.item())

        
class ogbg_molhiv(RealDataset):  

    def load_info(self):

        '''
        The following are the minimal information that define a dataset:
        - node_dim (int)
        - name (str)
        - size_qt005,size_qt095,size_qt050 (quantile of sizes of the graphs in the dataset)
        - default_alpha (float) default value of alpha
        '''

        self.node_dim = 50

        self.name = 'ogbg_molhiv'

        # Some precomputed values:

        self.size_min = 2
        self.size_max = 222
        self.size_qt005 = 11
        self.size_qt050 = 21
        self.size_qt095 = 45
        
        self.default_alpha = 0.5
        self.normalization = 0.11571682504362425

    def get_transform(self, normalize_w):

        self.atomencoder = AtomEncoder(emb_dim=50)
        for param in self.atomencoder.parameters():
            param.requires_grad = False
        self.atomencoder.load_state_dict(torch.load('saves/'+self.name+'_Data/atomencoder_50'))

        # Normalization for mean(gw) = mean(fGW)
        if normalize_w:
            unnormalized_gw = pd.read_csv('saves/'+self.name+'_Data/unnormalized_gw.csv')
            sigma = np.sqrt(unnormalized_gw['w'].mean()/unnormalized_gw['gw'].mean())
        else:
            sigma=1

        def transform(data):
            data.x = self.atomencoder(data.x)/sigma
            data.edge_index = add_remaining_self_loops(data.edge_index)[0]
            return data
        
        return transform
    
    def load_dataset(self):
        print('Note: using transform, not pretransform for avoiding error but this is slow. TO DO!')
        dataset = PygGraphPropPredDataset(name = "ogbg-molhiv", root = 'saves/'+self.name+'_Data', transform=self.transform)
        return dataset   
    
    
    def define_metric(self,custom_metric,alpha):

        if alpha == 'auto' and custom_metric == False:
            self.alpha = self.default_alpha
            self.metric = fgw_Metric(alpha)
        else:
            # TO DO
            pass

        self.feature = ogb_label()

    def get_dataloader_reg(self,batchsize = 512, TSNE = False, dataset_size = 100000):
        if not(TSNE):
            return DataLoader([(G,self.feature(G)) for G in self.data[:dataset_size]],batch_size = batchsize,shuffle = True)
        else:
            dataset_1 = [(G,self.feature(G)) for G in self.data if G.y.item()==1]
            dataset_0 = [(G,self.feature(G)) for G in self.data[:len(dataset_1)]]
            return DataLoader(dataset_1+dataset_0,batch_size = batchsize, shuffle = True)
       

if __name__ == '__main__':
    
    #import torch
    #from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
    #atom_encoder = AtomEncoder(emb_dim = 50)
    #torch.save(atom_encoder.state_dict(),'saves/ogbg_molhiv_Data/atomencoder_50')

    pass