import torch 
from datasets.base import Dataset
from .loss import to_FCh_format
import matplotlib.pyplot as plt
from ot.gromov import gromov_wasserstein
import numpy as np


def plot_recons(encoder: torch.nn.Module,
                decoder: torch.nn.Module,
                dataset: Dataset,
                n_recons: int,
                device = 'cuda',
                save_dir = None
                ):
    
    encoder.to(device)
    decoder.to(device)

    encoder.eval()
    decoder.eval()
    
    dataloader = dataset.get_dataloader(batchsize = n_recons, dataset_size=n_recons)

    for inputs in dataloader:

        inputs = inputs.to(device)

        outputs_F, outputs_C, outputs_h = decoder(encoder(inputs))
        outputs = list(zip(outputs_F,outputs_C,outputs_h))

        targets_F, targets_C, targets_h = to_FCh_format(inputs, device = device )
        targets = list(zip(targets_F,targets_C,targets_h))

        k = 0
        for output,target in zip(outputs,targets):

            X,C,h = output
            X_pred,C_pred,h_pred = X.detach().cpu(),C.detach().cpu(),h.detach().cpu()

            X,C,h = target
            X_trgt,C_trgt,h_trgt = X.detach().cpu(),C.detach().cpu(),h.detach().cpu()

            #### MATCHING ####

            gw0 = gromov_wasserstein(C_pred, C_trgt, h_pred, h_trgt, 'square_loss', verbose=False, log=False)

            map=np.argmax(gw0,axis=1)
            permutation=np.argsort(map)

            X_pred = X_pred[permutation]
            C_pred = C_pred[permutation][:, permutation]
            h_pred = h_pred[permutation]
    
            #### MATCHING DONE ####


            fig, axes = plt.subplots(2,2, figsize = (16,16))

            dataset.plot(axes[0,0],axes[0,1],X_pred,C_pred,h_pred)

            dataset.plot(axes[1,0],axes[1,1],X_trgt,C_trgt,h_trgt)

            plt.show()
            fig.savefig(f'plots/recons{k}.pdf',bbox_inches='tight')
            if save_dir!=None:
                fig.savefig(save_dir+f'/recons{k}.pdf',bbox_inches='tight')
            k+=1
        

        break

     