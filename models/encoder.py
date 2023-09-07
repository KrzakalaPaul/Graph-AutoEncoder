import torch
from torch.nn import Linear,Embedding
from torch_geometric.nn.aggr import SumAggregation
from torch_geometric.nn.models import GIN 
from torch_geometric.data import Data,Batch


class Basic_Encoder(torch.nn.Module):

    def __init__(self,in_channels: int,
                      hidden_channels: int,
                      embedding_channels: int,
                      depth: int, # Depth = Number of hidden layers
                      dropout = 0.,
                      **kwargs
                      ):

        super().__init__()
        
        self.depth = depth

        self.params = {'in_channels':in_channels,
                       'hidden_channels':hidden_channels,
                       'embedding_channels':embedding_channels,
                       'dropout':dropout,
                       'depth':depth, 
                       'name':'basic_encoder'}

        self.embedding_channels = embedding_channels

        self.model=GIN(in_channels =  in_channels, 
                       hidden_channels =  hidden_channels,
                       num_layers = depth + 1, 
                       out_channels = hidden_channels,
                       dropout = dropout)
        
            
        self.aggregation_layer=SumAggregation()

        self.linear=Linear(in_features=hidden_channels,out_features=embedding_channels)
        
    def get_grads(self):
        list_grads =  []
        for k in range(self.depth):
            W = self.model.convs[k].nn.lins[0].weight
            list_grads.append(W.grad)
        return list_grads
    
    def get_weight(self):
        return self.model.convs[-1].nn.lins[0].weight

    
    def forward(self, data: Data):

        x, edge_index = data.x, data.edge_index

        x = self.model(x,edge_index)

        global_x = self.aggregation_layer(x,data.batch)

        global_x = self.linear( global_x.relu() )
           
        return global_x

    

# ----------------------------------------------------- #

from torch_geometric.utils import softmax

class GatedAggregation(torch.nn.Module):

    def __init__(self,in_features,out_features):
        super().__init__()
        self.h=Linear(in_features=in_features,out_features=out_features)
        #self.gate=Linear(in_features=in_features,out_features=1)
        self.gate=Linear(in_features=in_features,out_features=in_features)
        self.aggr=SumAggregation()

    def forward(self,x,batch): 
        x = self.h(x)*softmax(self.gate(x),batch)
        return self.aggr(x,batch)
    

class Gated_Encoder(torch.nn.Module):

    def __init__(self,in_channels: int,
                      hidden_channels: int,
                      embedding_channels: int,
                      depth: int, # Depth = Number of hidden layers
                      dropout = 0.,
                      **kwargs
                      ):

        super().__init__()

        self.depth = depth

        self.params = {'in_channels':in_channels,
                       'hidden_channels':hidden_channels,
                       'embedding_channels':embedding_channels,
                       'dropout':dropout,
                       'depth':depth,
                       'name':'gated_encoder'}
        
        self.embedding_channels = embedding_channels

        self.model=GIN(in_channels =  in_channels, 
                       hidden_channels =  hidden_channels,
                       num_layers = depth + 1,  
                       out_channels = hidden_channels,
                       dropout = dropout)
        
            
        self.aggregation_layer=GatedAggregation(in_features =  hidden_channels,
                                                out_features = hidden_channels)

        self.linear=Linear(in_features=hidden_channels,out_features=embedding_channels)
    
    def get_grads(self):
        list_grads =  []
        for k in range(self.depth):
            W = self.model.convs[k].nn.lins[0].weight
            list_grads.append(W.grad)
        return list_grads
    
    def get_weight(self):
        return self.model.convs[-1].nn.lins[0].weight

    def forward(self, data: Data):

        x, edge_index = data.x, data.edge_index

        x = self.model(x,edge_index)

        global_x = self.aggregation_layer(x,data.batch)

        global_x = self.linear( global_x.relu() )
           
        return global_x
    



# ----------------------------------------------------- #

from torch_geometric.utils import scatter,unbatch,unbatch_edge_index
from torch_geometric.transforms import VirtualNode



class Virtual_Encoder(torch.nn.Module):

    def __init__(self,in_channels: int,
                      hidden_channels: int,
                      embedding_channels: int,
                      depth: int, # Depth = Number of hidden layers
                      dropout = 0.,
                      **kwargs
                      ):

        super().__init__()

        self.depth = depth

        self.params = {'in_channels':in_channels,
                       'hidden_channels':hidden_channels,
                       'embedding_channels':embedding_channels,
                       'dropout':dropout,
                       'depth':depth,
                       'name':'virtual_encoder'}
        
        self.embedding_channels = embedding_channels

        self.transform = VirtualNode()

        self.model=GIN(in_channels =  in_channels, 
                       hidden_channels =  hidden_channels,
                       num_layers = depth + 1,  
                       out_channels = hidden_channels,
                       dropout = dropout)


        self.linear = Linear(in_features=embedding_channels,out_features=embedding_channels)
        self.aggr = SumAggregation()


    def add_virtual_node(self,batch):

        x_list = unbatch(src = batch.x, batch = batch.batch)
        edge_index_list = unbatch_edge_index(edge_index=batch.edge_index,batch=batch.batch)

        new_data_list = []
        for x,edge_index in zip(x_list,edge_index_list):

            data = Data(x=x, edge_index=edge_index)
            data = self.transform(data)
            data.mask = torch.zeros(len(data.x),dtype = torch.bool,device=data.x.device)
            data.mask[-1] = 1 
            new_data_list.append(data)

        batch = Batch.from_data_list(new_data_list)
        
        return batch
    
    def get_grads(self):
        list_grads =  []
        for k in range(self.depth):
            W = self.model.convs[k].nn.lins[0].weight
            list_grads.append(W.grad)
        return list_grads
    
    def get_weight(self):
        return self.model.convs[-1].nn.lins[0].weight

    def forward(self, batch: Data):

        batch = self.add_virtual_node(batch)

        x, edge_index = batch.x, batch.edge_index

        x = self.model(x,edge_index)

        # mask everything except virtual nodes

        x = x*batch.mask.reshape(-1,1)

        global_x = self.aggr(x,batch.batch)

        global_x = self.linear( global_x.relu() )
           
        return global_x