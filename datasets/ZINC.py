

from .base import RealDataset,Feature,fgw_Metric
from .utils import plot_adjacency
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.datasets import ZINC as torch_geometric_ZINC

from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import torch 
import numpy as np
import pandas as pd



class zinc_label(Feature):

    def __init__(self):
        self.name = 'penalized logP'
        self.type = 'continuous' # ='discrete'
        self.scoring = 'neg_mean_absolute_error'
        self.dim = 1

    def __call__(self,G : Data):
        return torch.tensor([G.y.item()])
        
class ZINC(RealDataset):  

    def load_info(self):

        '''
        The following are the minimal information that define a dataset:
        - node_dim (int)
        - name (str)
        - size_qt005,size_qt095,size_qt050 (quantile of sizes of the graphs in the dataset)
        - default_alpha (float) default value of alpha
        '''

        self.node_dim = 50

        self.name = 'ZINC'

        # Some precomputed values:

        self.size_min = None
        self.size_max = None
        self.size_qt005 = 16
        self.size_qt050 = 23
        self.size_qt095 = 31
        
        self.default_alpha = 0.7
        self.normalization = 0.10141294331222771


    def get_transform(self, normalize_w: bool):

        embedding_layer = torch.nn.Embedding(28, 50)
        for param in embedding_layer.parameters():
            param.requires_grad = False
        embedding_layer.load_state_dict(torch.load('saves/'+self.name+'_Data/node_embedding'))

        if normalize_w: # Normalization for mean(gw) = mean(fGW)
            unnormalized_gw = pd.read_csv('saves/'+self.name+'_Data/unnormalized_gw.csv')
            sigma = np.sqrt(unnormalized_gw['w'].mean()/unnormalized_gw['gw'].mean())
        else:
            sigma=1

        def transform(data: Data):
            data.x = embedding_layer(data.x).squeeze()/sigma
            data.edge_index = add_remaining_self_loops(data.edge_index)[0]
            return data
        
        return transform
    
    def load_dataset(self):
        print('Note: using transform, not pretransform for avoiding error but this is slow. TO DO!')
        dataset = torch_geometric_ZINC(root = 'saves/'+self.name+'_Data', transform=self.transform)
        return dataset   
    
    
    def define_metric(self,custom_metric,alpha):

        if alpha == 'auto' and custom_metric == False:
            self.alpha = self.default_alpha
            self.metric = fgw_Metric(alpha)
        else:
            # TO DO
            pass
        self.feature = zinc_label()

    def get_dataloader_reg(self,batchsize = 512, TSNE = False, dataset_size = 100000):
        if not(TSNE):
            return DataLoader([(G,self.feature(G)) for G in self.data[:dataset_size]],batch_size = batchsize,shuffle = True)
        else:
            dataset = [(G,self.feature(G)) for G in self.data if G.y.item()>-6]
            return DataLoader(dataset,batch_size = batchsize, shuffle = True)
    
    def plot(self,ax1,ax2,X,C,h):

        ax1.scatter(X[:,0],X[:,1])
        plot_adjacency(C,h,ax2)


if __name__ == '__main__':

    #embedding_layer = torch.nn.Embedding(28, 50)
    #torch.nn.init.xavier_uniform_(embedding_layer.weight.data)
    #torch.save(embedding_layer.state_dict(),'saves/ZINC_Data/node_embedding')

    dataset = ZINC().data
    y = [G.y.item() for G in dataset[:10000]]
    y = np.array(y)

    print('Quantile 1%')
    print(np.quantile(y,0.01))
    print('Quantile 99%')
    print(np.quantile(y,0.99))
