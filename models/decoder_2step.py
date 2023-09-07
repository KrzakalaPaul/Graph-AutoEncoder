import torch
from torch_geometric.nn.models import MLP
from torch.nn.functional import softmax
from torch.nn import Unflatten

class adjacency_NN(torch.nn.Module):

    def __init__(self,N_max: int,
                      in_channels = 2,
                      edge_embedding_channels = 10,
                      dropout = 0.,
                      num_layers = 3
                      ):

        super().__init__()
        
        self.get_euclidian=MLP(in_channels=in_channels,
                               hidden_channels=N_max*edge_embedding_channels,
                               out_channels=N_max*edge_embedding_channels,
                               dropout=dropout,
                               num_layers=num_layers)
        
        self.unflatten = Unflatten(dim = 1, unflattened_size=(N_max,edge_embedding_channels))

        self.mask = torch.eye(N_max,dtype=torch.bool).reshape((1, N_max, N_max))

    
    def forward(self,input):

        batchsize=input.shape[0]

        x = self.get_euclidian(input)
        x = self.unflatten(x)

        # Dot product 
        dot=torch.einsum('ikd,ild->ikl', x, x)
        C=torch.sigmoid(dot)

        mask = self.mask.to(C.device).repeat(batchsize, 1, 1)
        C=C*(~mask)+mask # ADD SELF LOOPS

        return C


class Decoder2STEP(torch.nn.Module):

    def __init__(self,nodes_channels = 2,
                      embedding_channels = 10,
                      dropout = 0.,
                      N_max = 20,
                      edge_embedding_channels = 5,
                      num_layers = 1,
                       **kwargs
                      ):
        
        self.nodes_channels = nodes_channels
        self.N_max = N_max
        
        self.params = {'nodes_channels':nodes_channels,
                       'embedding_channels':embedding_channels,
                       'dropout':dropout,
                       'N_max':N_max,
                       'edge_embedding_channels':edge_embedding_channels,
                       'name':'MLP_decoder'}
        
        super().__init__()

        self.C_net = adjacency_NN(in_channels=embedding_channels,
                                  edge_embedding_channels=edge_embedding_channels,
                                  N_max=N_max,
                                  dropout=dropout,
                                  num_layers=num_layers)
        

        self.F_net = MLP(in_channels=embedding_channels,
                         hidden_channels=N_max*nodes_channels,
                         out_channels=N_max*nodes_channels,
                         dropout=dropout,
                         num_layers=num_layers)
        
        self.F_unflatten = Unflatten(dim = 1, unflattened_size=(N_max,nodes_channels))
        
        self.h_net = MLP(in_channels=embedding_channels,
                         hidden_channels=N_max,
                         out_channels=N_max,
                         dropout=dropout,
                         num_layers=num_layers)

    
    def forward(self,embedding):

        F_batch=self.F_unflatten(self.F_net(embedding))
        C_batch=self.C_net(embedding)
        h_batch=softmax(self.h_net(embedding),dim=1)

        return F_batch,C_batch,h_batch
    
