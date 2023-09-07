from datasets.base import Dataset
import torch as torch
from torch_geometric.data import Batch
from torch.nn import MSELoss,Module
from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt
import wandb
from math import sqrt

def metric_learning_eval(encoder: Module,
                         dataset: Dataset,
                         batchsize = 512,
                         n_samples = 1000,
                         device = 'cuda',
                         save_dir = None
                         ):
    
    size = 0

    print('Preparing Dataloader....')
    dataloader = dataset.get_dataloader_ml(batchsize=batchsize)
    print('Done')

    target_dists=[]
    pred_dists=[]

    encoder.to(device)
    encoder.eval()

    loss_fn = MSELoss(reduction='sum')
    loss = 0

    while size < n_samples:

        for inputs1,inputs2,targets in dataloader: 

            inputs1 = inputs1.to(device)
            inputs2 = inputs2.to(device)

            print('')
            print(encoder(inputs1)[0])

            #outputs = (torch.linalg.vector_norm(encoder(inputs1) - encoder(inputs2),dim=1)**2).detach().cpu().numpy()
            outputs = torch.sum((encoder(inputs1) - encoder(inputs2))**2,dim=1).detach().cpu()

            target_dists += list(torch.sqrt(torch.abs(targets))) # abs for numerical issues
            pred_dists += list(torch.sqrt(torch.abs(outputs)))

            loss += loss_fn(outputs,targets).item()

            size+=batchsize

            if size>n_samples:
                break

    loss /= size
    print(f'loss : {loss}')

    target_dists = np.array(target_dists)
    pred_dists = np.array(pred_dists)

    corr = np.vstack([target_dists,pred_dists])
    corr = np.corrcoef(corr)[0,1]
    print(f'correlation : {corr}')

    fig, ax = plt.subplots()
    plt.scatter (target_dists,pred_dists,alpha=0.5)
    ax.set_title(f'Correlation: {corr}')
    ax.set_xlabel(f'Input dist ({dataset.metric.name})')
    ax.set_ylabel('Embedding dist (Euclidian)')

    m = max(np.max(target_dists),np.max(pred_dists))
    ax.set_xlim([0, m])
    ax.set_ylim([0, m])

    plt.show()
    fig.savefig(f'plots/metric_learning.pdf',bbox_inches='tight')
    if save_dir!=None:
        fig.savefig(save_dir+f'/metric_learning.pdf',bbox_inches='tight')



from time import time

def metric_learning(encoder: Module, 
                    dataset: Dataset,
                    batchsize = 512,
                    n_gradients = 100,
                    device = 'cuda'
                    ):
    
    print('Preparing dataloader ...')
    dataloader = dataset.get_dataloader_ml(batchsize=batchsize)
    print('Done')

    loss_fn = torch.nn.HuberLoss(delta=1.0)
    optimizer = Adam(params=encoder.parameters())

    encoder.to(device)
    encoder.train()

    d = encoder.params['embedding_channels']

    g = 0

    while g < n_gradients:

        for inputs1,inputs2,targets in dataloader: 

            # Get Batch

            inputs1 = inputs1.to(device)
            inputs2 = inputs2.to(device)
            targets = targets.to(device)

            print(inputs1.x.shape)
            
            # Get pred 

            #outputs = torch.linalg.vector_norm(encoder(inputs1) - encoder(inputs2),dim=1)**2

            outputs = torch.sum((encoder(inputs1) - encoder(inputs2))**2,dim=1)

            # Compute loss

            #loss = loss_fn(outputs/sqrt(d),targets/sqrt(d))
            loss = loss_fn(outputs,targets)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            clipping_value = 10 # arbitrary value of your choosing
            torch.nn.utils.clip_grad_norm(encoder.parameters(), clipping_value)
            optimizer.step()
            

            print(loss.item())

            g+=1

            if g > n_gradients:
                break

            log_dic = {'gradients': g,
                       'loss': loss.item()}
            
            grads_encoder = encoder.get_grads()
            for k,grad in enumerate(grads_encoder):
                log_dic[f'Encoder layer {k} (grad inf-norm)'] = torch.abs(grad).max()
            
            if g%100==0:
                print(log_dic)
            wandb.log(log_dic)
 
