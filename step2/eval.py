import torch 
import torch_geometric
from torch.utils.data import dataloader
from datasets.base import Dataset
from .loss import to_FCh_format,fgw_loss,w_loss
import pandas as pd

def eval(encoder: torch.nn.Module,
         decoder: torch.nn.Module,
         dataset: Dataset = None,
         batchsize: int = 512,
         n_samples: int = 10000,
         loss_fun_gw = 'square_loss',
         loader_eval: dataloader = None,
         alpha:float = None,
         normalization: float = None,
         device = 'cuda',
         save_dir = None
         ):
    
    '''
    # Can be given either dataset/n_samples/batchsize
    # Or directly a dataloader/alpha/normalizatio
    '''
    
    encoder.to(device)
    decoder.to(device)

    encoder.eval()
    decoder.eval()

    if loader_eval == None:
        alpha = dataset.alpha
        normalization = dataset.normalization
        loader_eval = dataset.get_dataloader(batchsize=batchsize,dataset_size=n_samples)

    loss_fgw = 0
    loss_w = 0
    size = 0

    while size<n_samples:

        for inputs in loader_eval:

            batchsize = len(inputs)
            size += batchsize

            inputs = inputs.to(device)

            outputs_F, outputs_C, outputs_h = decoder(encoder(inputs))
            outputs = list(zip(outputs_F,outputs_C,outputs_h))

            targets_F, targets_C, targets_h = to_FCh_format(inputs,device=device)
            targets = list(zip(targets_F,targets_C,targets_h))

            

            for output,target in zip(outputs,targets):
                loss_fgw += fgw_loss(output,target,alpha=alpha,loss_fun_gw=loss_fun_gw).item()
                loss_w += w_loss(output,target).item()

            '''
            print(f" Loss : {loss}")

            loss = 0
            targets.reverse()
            for output,target in zip(outputs,targets):
                loss += fgw_loss(output,target,alpha=alpha,loss_fun_gw=loss_fun_gw)

            loss /= batchsize

            print(f" Loss rd inputs: {loss}")

            '''

            if size>n_samples:
                break

    loss_fgw /= size
    loss_w /= size


    if normalization == None:
        if save_dir!=None:
            pd.Series({'loss_w': loss_w, 'loss_fgw': loss_fgw}).to_csv(save_dir+'/loss.csv')
        print({'loss_w': loss_w, 'loss_fgw': loss_fgw})
        return loss_fgw,loss_w
    else:
        if save_dir!=None:
            pd.Series({'loss_w': loss_w, 'loss_fgw': loss_fgw, 'loss_fgw_normalized': loss_fgw/normalization}).to_csv(save_dir+'/loss.csv')
        print({'loss_w': loss_w, 'loss_fgw': loss_fgw, 'loss_fgw_normalized': loss_fgw/normalization})
        return loss_fgw/normalization,loss_w/normalization
    