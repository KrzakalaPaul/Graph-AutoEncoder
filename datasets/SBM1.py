
from .SBM2 import Instance_SBM2,SBM2,Feature
import torch 
import numpy as np
import networkx as nx

from .utils import plot_adjacency


size_min = 3
size_max = 9

class Instance_SBM1(Instance_SBM2):

    def __init__(self,p=0.9,q=0.1):

        sizes=[ np.random.randint(size_min,size_max+1), np.random.randint(size_min,size_max+1)]
        centroid1 = np.random.normal(loc=0,scale=1,size=2)
        centroid1 = centroid1/np.linalg.norm(centroid1)

        isclose = True
        while isclose:
            centroid2 = np.random.normal(loc=0,scale=1,size=2)
            centroid2 = centroid2/np.linalg.norm(centroid2)
            isclose = np.arccos(np.dot(centroid1,centroid2))<np.pi/2
            
        proba_mat = np.array([[p,q],[q,p]])
        
        self.G = nx.generators.community.stochastic_block_model(sizes,proba_mat,selfloops=False)
        self.x = np.vstack([ centroid1.reshape(1,-1) for _ in range(sizes[0])] + [ centroid2.reshape(1,-1) for _ in range(sizes[1])])
        self.x = torch.tensor(self.x,dtype=torch.float32)
        self.x +=  torch.randn(self.x.size()) * 0.1

        self.sizes = sizes

class Feature_SBM1(Feature):

    def __init__(self):
        self.name = 'Size clusters'
        self.type = 'continuous' # ='discrete'
        self.dim = 2

    def __call__(self,x: Instance_SBM1):
        small_size = min([x.sizes[0],x.sizes[1]])
        big_size = max([x.sizes[0],x.sizes[1]])
        y = torch.tensor([small_size,big_size],dtype=float)
        return y


class SBM1(SBM2):


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

        self.name = 'SBM1'

        self.size_qt005 = 7
        self.size_qt095 = 17
        self.size_qt050 = 12

        self.default_alpha = 0.6 # Default alpha for normalized_w = True using my maximization method
        self.normalization = 0.18838

      
    def get_instance(self):
        return Instance_SBM1()
    
    def define_metric(self,custom_metric,alpha):

        super().define_metric(custom_metric,alpha)

        self.feature = Feature_SBM1()


    def plot(self,ax1,ax2,X,C,h):

        ax1.scatter(X[:,0],X[:,1])
        ax1.set_xbound(lower=-2, upper=2)
        ax1.set_ybound(lower=-2, upper=2)

        plot_adjacency(C,h,ax2)

        #N1 = torch.sum(torch.where(z>0,0,1))
        
