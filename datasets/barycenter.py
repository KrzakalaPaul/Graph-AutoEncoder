
from torch_geometric.data import Data
from random import choice
import numpy as np
from ot.gromov import fgw_barycenters,fused_gromov_wasserstein2
from ot.utils import unif,dist
import torch 
from scipy.sparse import coo_matrix
import networkx as nx
import matplotlib.pyplot as plt 
from torch_geometric.loader import DataLoader
import pandas as pd

size_min = 5
size_max = 15

class Instance():

    def __init__(self,Ys,Cs,ps,save = None):

        if save==None:
            self.weights = torch.rand(3)
            self.weights /= torch.sum(self.weights)
            p = torch.tensor(unif(N))

            X,C = fgw_barycenters(N, Ys, Cs, ps, self.weights, 0.5, p=p, loss_fun='square_loss', max_iter = 50)

            self.X = X.float()
            self.A = torch.where(C>torch.rand_like(C),torch.ones_like(C),torch.zeros_like(C))
        else:
            self.weights = save["w"]
            self.X = save["X"]
            self.A = save["A"]

    def to_torch_data(self):
        
        adjacency=coo_matrix(self.A)

        row=list(adjacency.row)
        col=list(adjacency.col)

        # ADD SELF LOOPS
        for i in range(N):
            row.append(i)
            col.append(i)

        row=torch.tensor(row,dtype=torch.long).reshape(-1,1)
        col=torch.tensor(col,dtype=torch.long).reshape(-1,1)

        edge_index=torch.hstack((row,col))

        data=Data(x=self.X,edge_index=edge_index.t())

        return data


class Feature():

    def __init__(self):
        self.name = 'Weights'
        self.type = 'continuous' # ='discrete'
        self.dim = 3

    def __call__(self,x: Instance):
        return x.weights
        
    
class Metric():

    def __init__(self,fGW = False):
        self.fGW = fGW
        if self.fGW:
            self.name = '0.5-fGW'
        else:
            self.name = 'L1 between weights'

    def __call__(self,x: Instance,y: Instance):
        return torch.sum(torch.abs(x.weights-y.weights))
    
 
class Dataset():

    def __init__(self,fGW = False):
        
        self.metric = Metric(fGW = fGW)
        self.feature = Feature()
        self.node_dim = 2
        self.fGW = fGW

        # Get Templates

        self.templates = [] 
        
        # Circle With Same feature
        A1 = torch.tensor(np.eye(N,k=1) + np.eye(N,k=-1), dtype=torch.float64)
        A1[0,-1] += 1 
        A1[-1,0] += 1 
        T1 = (torch.ones(N,2,dtype=torch.float64),A1)

        # Line with alternating features
        x2 = torch.empty(N,2, dtype=torch.float64)
        for k in range(N):
            if k%2==0:
                x2[k] = torch.ones(2, dtype=torch.float64)
            if k%2==1:
                x2[k] = -torch.ones(2, dtype=torch.float64)

        A2 = torch.tensor(np.eye(N,k=1) + np.eye(N,k=-1), dtype=torch.float64)
        T2 = (x2,A2)

        # SBM
        x3 = torch.vstack([torch.ones(5,2, dtype=torch.float64), -torch.ones(5,2, dtype=torch.float64) ])
        A3 = torch.block_diag(torch.ones(5,5, dtype=torch.float64),torch.ones(5,5, dtype=torch.float64))

        T3 = (x3,A3)

        self.Ys = [T1[0],T2[0],T3[0]]
        self.Cs = [T1[1],T2[1],T3[1]]
        self.ps = [(torch.tensor(unif(N))) for _ in range(3)]

        # Load precomputed barycenters

        A = torch.load(f'saves/Barycenter_Data/A.pt')
        X = torch.load(f'saves/Barycenter_Data/X.pt')
        w = torch.load(f'saves/Barycenter_Data/w.pt')

        self.data = [Instance(self.Ys,self.Cs,self.ps,save = {'A':A[k],'X': X[k], 'w': w[k]}) for k in range(n_samples)]
        self.fGW_dataframe = pd.read_csv('saves/Barycenter_Data/fgw.csv')
        self.n_samples = n_samples

        #self.normalization = self.fGW_dataframe['fgw0.5'].mean()
        self.normalization = 0.276
        self.size_qt005 = 10
        self.size_qt095 = 10
        self.size_qt050 = 10

    def get_instance(self,from_save = True):
        if not(from_save):
            return Instance(self.Ys,self.Cs,self.ps,save=None)
        else:
            k = np.random.randint(0,n_samples)
            return self.data[k]
    
    def get_dataloader(self,batchsize = 512,train = True):
        return DataLoader([I.to_torch_data() for I in self.data],batch_size = batchsize,shuffle = True)
    
    def get_dataloader_reg(self,batchsize = 512,train = True, TSNE = False):
        return DataLoader([(I.to_torch_data(),self.feature(I)) for I in self.data],batch_size = batchsize,shuffle = True)
    
    def get_dataloader_ml(self,batchsize = 512,train = True):
        if self.fGW:
            dataframe = self.fGW_dataframe
            return DataLoader([(self.data[int(row['i'])].to_torch_data(),
                                self.data[int(row['j'])].to_torch_data(),
                                row['fgw0.5'])
                                #np.sqrt(np.abs(row['fgw0.5'])))
                                for index, row in dataframe.iterrows()],batch_size = batchsize,shuffle = True)
        else:
            data_ml = self.data[:int(np.sqrt(self.n_samples))]
            return DataLoader([(I1.to_torch_data(),I2.to_torch_data(),self.metric(I1,I2)) for I1 in data_ml for I2 in data_ml],batch_size = batchsize,shuffle = True)
    


if __name__ == "__main__":
    '''
    As = []
    Xs = []
    ws = []

    for k in range(n_samples):
        I = Dataset().get_instance(from_save = False)

        A = I.A
        X = I.X
        w = I.weights

        As.append(A.unsqueeze(0))
        Xs.append(X.unsqueeze(0))
        ws.append(w.unsqueeze(0))

    A = torch.concat(As)
    X = torch.concat(Xs)
    w = torch.concat(ws)

    torch.save(A, f'saves/Barycenter_Data/A.pt')
    torch.save(X, f'saves/Barycenter_Data/X.pt')
    torch.save(w, f'saves/Barycenter_Data/w.pt')
    

    dataset = Dataset()

    i_list = []
    j_list = []
    fgw_list = []

    for i,j in zip(np.random.randint(0,len(dataset.data),n_fgw),np.random.randint(0,len(dataset.data),n_fgw)):
     
        I1 = dataset.data[i]
        I2 = dataset.data[j]

        M = dist(I1.X,I2.X)
        p1 = torch.tensor(unif(N))
        p2 = torch.tensor(unif(N))
        C1 = I1.A
        C2 = I2.A

        fgw = fused_gromov_wasserstein2(M,C1=C1,C2=C2,p=p1,q=p2,alpha=0.5)

        
        i_list.append(i)
        j_list.append(j)
        fgw_list.append(fgw.item())

    dataframe = pd.DataFrame({'i': i_list,'j': j_list,'fgw0.5': fgw_list})
    dataframe.to_csv('saves/Barycenter_Data/fgw.csv')

    '''

