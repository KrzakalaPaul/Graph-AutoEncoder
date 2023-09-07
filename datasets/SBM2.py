
from .base import Instance,SyntheticDataset,Feature,Custom_Metric,fgw_Metric

from torch_geometric.data import Data
import torch 
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt 
from .utils import plot_adjacency


size_min = 3
size_max = 9

class Instance_SBM2(Instance):

    def __init__(self,p=0.9,q=0.1):

        sizes=[ np.random.randint(size_min,size_max+1), np.random.randint(size_min,size_max+1)]
        proba_mat = np.array([[p,q],[q,p]])
        
        self.G = nx.generators.community.stochastic_block_model(sizes,proba_mat,selfloops=False)
        self.x = torch.concat([ -torch.ones((sizes[0],2)),torch.ones((sizes[1],2))])
        self.x +=  torch.randn(self.x.size()) * 0.1

        self.sizes = sizes

    def to_torch_data(self):

        adjacency=nx.to_scipy_sparse_array(self.G,format='coo',weight=None)

        row=list(adjacency.row)
        col=list(adjacency.col)

        # ADD SELF LOOPS
        for i in range(len(self.G)):
            row.append(i)
            col.append(i)

        row=torch.tensor(row,dtype=torch.long).reshape(-1,1)
        col=torch.tensor(col,dtype=torch.long).reshape(-1,1)

        edge_index=torch.hstack((row,col))

        data=Data(x=self.x,edge_index=edge_index.t(),sizes = self.sizes)

        return data

    def display(self):
        plt.figure()
        nx.draw(self.G)
        plt.show()

'''
class Feature_SBM2(Feature):

    def __init__(self):
        self.name = 'Size cluster 1'
        self.type = 'discrete' # ='continuous'
        self.dim = size_max+1 - size_min

    def __call__(self,x: Instance):
        y = torch.zeros(self.dim,dtype=float)
        y[x.sizes[0]-size_min]=1.
        return y
    
'''
class Feature_SBM2(Feature):

    def __init__(self):
        self.name = 'Size clusters'
        self.type = 'continuous' # ='discrete'
        self.dim = 2

    def __call__(self,x: Instance):
        y = torch.tensor([x.sizes[0],x.sizes[1]],dtype=float)
        return y


class L1_Metric(Custom_Metric):

    def __init__(self):
        self.name = 'L1 between sizes'

    def __call__(self,x: Instance,y: Instance):
        return float(np.abs( x.sizes[0] - y.sizes[0]) + np.abs( x.sizes[1] - y.sizes[1]))


class SBM2(SyntheticDataset):


    def load_info(self):

        '''
        The following are the minimal information that define a dataset:
        - node_dim (int)
        - name (str)
        - size_qt005,size_qt095,size_qt050 (quantile of sizes of the graphs in the dataset)
        - default_alpha (float) default value of alpha
        '''

        self.node_dim = 2
        self.p = 0.9
        self.q = 0.1

        self.name = 'SBM2'

        self.size_qt005 = 7
        self.size_qt095 = 17
        self.size_qt050 = 12

        self.default_alpha = 0.8 # Default alpha for normalized_w = True using my maximization method
        self.normalization = 0.1963
        

    def define_metric(self,custom_metric,alpha):

        if custom_metric:
            self.metric = L1_Metric()
        
        else:
            if alpha == 'auto':
                alpha = self.default_alpha
            self.alpha = alpha
            self.metric = fgw_Metric(alpha)

        self.feature = Feature_SBM2()

    
    def get_instance(self):
        return Instance_SBM2()



    def plot(self,ax1,ax2,X,C,h):

        '''
        # Match by cluster
        z = torch.sum(X,dim=1)
        _, indices = torch.sort(z)
        
        X = X[indices]
        h = h[indices]
        C = C[indices][:,indices]
        '''

        ax1.scatter(X[:,0],X[:,1])
        ax1.set_xbound(lower=-2, upper=2)
        ax1.set_ybound(lower=-2, upper=2)

        plot_adjacency(C,h,ax2)

        #N1 = torch.sum(torch.where(z>0,0,1))
        

if __name__ == "__main__":


    '''
    sizes = []
    name = 'SBM2'
    for k in range(n_precomputed_samples):
        data = torch.load('saves/'+name+f'_Data/dataset/{k}')
        sizes.append(len(data.x))

    sizes = np.array(sizes)

    print(np.quantile(sizes,0.05))
    print(np.quantile(sizes,0.95))
    print(np.quantile(sizes,0.5))
    '''