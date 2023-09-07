
from .base import Instance,SyntheticDataset,Feature,Custom_Metric,fgw_Metric
from .utils import plot_adjacency

from torch_geometric.data import Data
import torch 
import numpy as np
from numpy.random import randint,choice
import networkx as nx
import matplotlib.pyplot as plt 
from torch_geometric.utils import unbatch,unbatch_edge_index,to_dense_adj

max_splits = 3
min_branch_size = 2
max_branch_size = 8

class Instance_Tree():

    def __init__(self):

        self.n_splits = randint(0,max_splits+1)
        self.branch_sizes = [ randint(min_branch_size,max_branch_size+1)  for _ in range(2*self.n_splits+1) ]
        self.branch_features = [np.random.uniform(low = -1,high = 1,size=2) for _ in range(2*self.n_splits+1)]

        children_dic = {}
        waiting_adoption = {0}

        for k in range(self.n_splits):
            parent = choice(list(waiting_adoption))

            waiting_adoption.remove(parent)
            waiting_adoption.add(2*k+1)
            waiting_adoption.add(2*k+2)
            children_dic[parent] = (2*k+1,2*k+2)

        self.children_dic = children_dic

    def to_torch_data(self):

        row = []
        col = []
        x = []

        N = sum(self.branch_sizes)

        branch = 0
        node = 0 
        queue_branch = []
        queue_node = []

        while True:
            
            x.append(self.branch_features[branch].reshape(1,-1))
            for _ in range(self.branch_sizes[branch]-1):
                # Connect within branch
                row.append(node)
                col.append(node+1)
                x.append(self.branch_features[branch].reshape(1,-1))
                node+=1
                

            if branch in self.children_dic:
                children = self.children_dic[branch]
                branch = children[0]
                queue_branch.append(children[1])
                queue_node.append(node)
                
                # Connect with child branch
                row.append(node)
                col.append(node+1)
                node+=1

                

            elif len(queue_branch)!=0:

                branch = queue_branch.pop()

                # Connect with child branch
                parent_node = queue_node.pop()
                row.append(parent_node)
                col.append(node+1)
                node+=1

            else:
                break

        x = np.vstack(x)
        x = torch.tensor(x,dtype=torch.float32)
        
        # Add symmetry
        sym_row = row + col
        sym_col = col + row

        # ADD SELF LOOPS
        row = sym_row + list(range(N))
        col = sym_col + list(range(N))
        
        row=torch.tensor(row,dtype=torch.long).reshape(-1,1)
        col=torch.tensor(col,dtype=torch.long).reshape(-1,1)

        edge_index=torch.hstack((row,col))

        data=Data(x=x,edge_index=edge_index.t())
        data.validate(raise_on_error=True)

        return data

class Feature_Tree(Feature):

    def __init__(self):
        self.name = 'Number of splits, Depth of Branch 0'
        self.type = 'continuous' # ='discrete'
        self.dim = 2

    def __call__(self,x:Instance_Tree):
        y = torch.tensor([x.n_splits,x.branch_sizes[0]],dtype=float)
        return y
    


class TREES(SyntheticDataset):


    def load_info(self):

        '''
        The following are the minimal information that define a dataset:
        - node_dim (int)
        - name (str)
        - size_qt005,size_qt095,size_qt050 (quantile of sizes of the graphs in the dataset)
        - default_alpha (float) default value of alpha
        '''

        self.node_dim = 2

        self.name = 'TREES'

        self.size_qt005 = 19
        self.size_qt095 = 36
        self.size_qt050 = 28

        self.default_alpha = 0.6 # Default alpha for normalized_w = True using my maximization method
        self.normalization = 0.048
        

    def define_metric(self,custom_metric,alpha):

        if custom_metric:
            pass
        
        else:
            if alpha == 'auto':
                alpha = self.default_alpha
            self.alpha = alpha
            self.metric = fgw_Metric(alpha)

        self.feature = Feature_Tree()

    
    def get_instance(self):
        return Instance_Tree()



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
        ax1.set_xbound(lower=-1.1, upper=1.1)
        ax1.set_ybound(lower=-1.1, upper=1.1)

        plot_adjacency(C,h,ax2)

        #N1 = torch.sum(torch.where(z>0,0,1))
        

if __name__ == "__main__":

    I = Instance_Tree()
    data = I.to_torch_data()

    C = to_dense_adj(edge_index = data.edge_index).squeeze(0)
    x = data.x

    '''
    print(C)
    plt.figure()
    plt.imshow(C)
    plt.show()
    '''

    plt.figure()
    ax = plt.gca()
    ax.scatter(x[:,0],x[:,1])
    ax.set_xbound(lower=-1.1, upper=1.1)
    ax.set_ybound(lower=-1.1, upper=1.1)
    plt.show()


    print(data.x)

    pass