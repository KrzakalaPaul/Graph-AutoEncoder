import torch
from torch_geometric.nn.models import MLP
from torch.nn.functional import softmax
from torch.nn import Unflatten,Flatten,Sequential
from torch.nn import TransformerEncoderLayer,TransformerEncoder,Linear

class DecoderTRANSFORMER(torch.nn.Module):

    def __init__(self,nodes_channels: int,
                      embedding_channels: int,
                      N_max: int,
                      edge_embedding_channels: int,
                      depth: int,
                      nhead: int,
                      d_transformer: int,
                      d_ffn: int,
                      dropout = 0.,
                      **kwargs
                      ):
        
        self.nodes_channels = nodes_channels
        self.N_max = N_max
        
        self.params = {'nodes_channels':nodes_channels,
                       'embedding_channels':embedding_channels,
                       'dropout':dropout,
                       'N_max':N_max,
                       'd_transformer':d_transformer,
                       'd_ffn':d_ffn,
                       'nhead':nhead,
                       'depth':depth,
                       'edge_embedding_channels':edge_embedding_channels,
                       'name':'TRANSFORMER_decoder'}
        
        super().__init__()
        
        self.pre_transformer = Sequential(MLP(in_channels=embedding_channels,hidden_channels=embedding_channels,out_channels=d_transformer*N_max,num_layers=1,dropout=dropout),
                                          Unflatten(dim = 1, unflattened_size = (N_max,d_transformer))
                                          )
        
        encoder_layer = TransformerEncoderLayer(d_model = d_transformer,
                                                nhead = nhead, 
                                                dim_feedforward = d_ffn, 
                                                dropout = dropout, 
                                                batch_first = True,
                                                norm_first  = True
                                                )

        self.transformer_stack = TransformerEncoder(encoder_layer,
                                                    num_layers = depth)


        self.Nodewise_Linear_F = Sequential(Flatten(start_dim=0,end_dim=1),
                                            Linear(in_features=d_transformer,out_features=nodes_channels),
                                            Unflatten(dim = 0, unflattened_size = (-1,N_max))
                                            )
        
        self.Nodewise_Linear_A = Sequential(Flatten(start_dim=0,end_dim=1),
                                            Linear(in_features=d_transformer,out_features=edge_embedding_channels),
                                            Unflatten(dim = 0, unflattened_size = (-1,N_max))
                                            )
        
        # For adding self-edges
        self.mask = torch.eye(N_max,dtype=torch.bool).reshape((1, N_max, N_max))
        
        
        self.Nodewise_Linear_h = Sequential(Flatten(start_dim=0,end_dim=1),
                                            Linear(in_features=d_transformer,out_features=1),
                                            Unflatten(dim = 0, unflattened_size = (-1,N_max))
                                            )
        

    def get_grads(self):
        list_grads =  []
        for k in range(len(self.transformer_stack.layers)):
            W = self.transformer_stack.layers[k].linear1.weight
            list_grads.append(W.grad)
        return list_grads
    
    def get_weight(self):
        return self.transformer_stack.layers[0].linear1.weight

    
    def forward(self,embedding):

        # Common part :

        x = self.pre_transformer(embedding)
        x = self.transformer_stack(x)

        # F head :
        F_batch = self.Nodewise_Linear_F(x)

        # C head :

        edge_embeddings = self.Nodewise_Linear_A(x)
        edges = torch.einsum('ikd,ild->ikl', edge_embeddings, edge_embeddings)
        C_batch = torch.sigmoid(edges)

        mask = self.mask.to(C_batch.device) 
        mask = mask.repeat(len(C_batch), 1, 1)
        C_batch=C_batch*(~mask)+mask # ADD SELF LOOPS

        # h head :

        h_batch=softmax(self.Nodewise_Linear_h(x).squeeze(),dim = 1)

        return F_batch,C_batch,h_batch
    
