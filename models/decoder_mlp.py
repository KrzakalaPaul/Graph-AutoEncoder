import torch
from torch_geometric.nn.models import MLP
from torch_geometric.nn import Linear
from torch.nn.functional import softmax
from torch.nn import Unflatten


class DecoderMLP(torch.nn.Module):

    def __init__(self,nodes_channels: int,
                      embedding_channels: int,
                      N_max: int,
                      edge_embedding_channels: int,
                      depth: int,
                      dropout = 0.,
                       **kwargs
                      ):
        
        self.nodes_channels = nodes_channels
        self.N_max = N_max
        self.depth = depth
        
        self.params = {'nodes_channels':nodes_channels,
                       'embedding_channels':embedding_channels,
                       'dropout':dropout,
                       'N_max':N_max,
                       'edge_embedding_channels':edge_embedding_channels,
                       'depth': depth,
                       'name':'MLP_decoder'}
        
        super().__init__()


        channel_list = [ int(2**(l/2)*embedding_channels) for l in range(depth + 2) ]
        common_channels = channel_list[-1]

        self.common_net = MLP(channel_list = channel_list,
                              dropout = dropout
                              )

        self.C_lin = Linear(in_channels = common_channels,
                            out_channels = N_max*edge_embedding_channels)
        self.C_unflatten = Unflatten(dim = 1, unflattened_size=(N_max,edge_embedding_channels))
        self.mask = torch.eye(N_max,dtype=torch.bool).reshape((1, N_max, N_max))

        self.F_lin = Linear(in_channels = common_channels,
                            out_channels = N_max*nodes_channels)
        self.F_unflatten = Unflatten(dim = 1, unflattened_size=(N_max,nodes_channels))
        
        self.h_lin = Linear(in_channels = common_channels,
                            out_channels = N_max)
        

    def get_grads(self):
        list_grads =  []
        for k in range(self.depth):
            W = self.common_net.lins[k].weight
            list_grads.append(W.grad)
        return list_grads

    def get_weight(self):
        return self.common_net.lins[0].weight
    
    def forward(self,embedding):

        batchsize=embedding.shape[0]

        # Common part :

        embedding = self.common_net(embedding)

        # F head :

        F_batch=self.F_unflatten(self.F_lin(embedding))

        # C head :

        edges_embedding=self.C_unflatten(self.C_lin(embedding))
        edges = torch.einsum('ikd,ild->ikl', edges_embedding, edges_embedding)
        C_batch=torch.sigmoid(edges)

        mask = self.mask.to(C_batch.device) 
        mask = mask.repeat(batchsize, 1, 1)
        C_batch=C_batch*(~mask)+mask # ADD SELF LOOPS

        # h head :

        h_batch=softmax(self.h_lin(embedding),dim=1)

        return F_batch,C_batch,h_batch
    

