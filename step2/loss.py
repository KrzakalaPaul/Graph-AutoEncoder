
from torch_geometric.utils import unbatch,unbatch_edge_index,to_dense_adj
from ot.utils import unif,dist
from ot.gromov import fused_gromov_wasserstein2
from ot import emd2
import torch 

def to_FCh_format(batch , device):

    '''
    Convert a batch of torch geometric graph 
    To a list of features matrices F, Adjacency matrices C, weights matrices h
    '''

    list_F = unbatch(batch.x,batch.batch)

    sparse_C_list = unbatch_edge_index(batch.edge_index,batch.batch)

    list_C = [ to_dense_adj(edge_index = spC, max_num_nodes = len(x)).squeeze(0) for spC,x in zip(sparse_C_list,list_F)  ]

    list_h = [ torch.tensor(unif(len(x)),device=device) for x in list_F]

    return list_F,list_C,list_h 


def fgw_loss(G_pred,G_target,alpha,loss_fun_gw):

    F1,C1,h1 = G_pred
    F2,C2,h2 = G_target

    h2 = h2.type(torch.float64)
    # Hack to ensure that up to float64 precision sum(h1) = sum(h2)
    with torch.no_grad():
        h2 = h1.type(torch.float64).sum()*h2/h2.sum()
    
    M = dist(F1,F2)

    return fused_gromov_wasserstein2(M=M,
                                     C1=C1,
                                     C2=C2,
                                     p=h1,
                                     q=h2,
                                     loss_fun=loss_fun_gw,
                                     alpha=alpha,
                                     max_iter=50)





def w_loss(G_pred,G_target):

    F1,C1,h1 = G_pred
    F2,C2,h2 = G_target

    h2 = h2.type(torch.float64)
    # Hack to ensure that up to float64 precision sum(h1) = sum(h2)
    with torch.no_grad():
        h2 = h1.type(torch.float64).sum()*h2/h2.sum()
    
    M = dist(F1,F2)

    return emd2(a = h1,
                b = h2,
                M = M)
