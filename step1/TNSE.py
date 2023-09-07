from datasets.base import Dataset
import torch as torch
from torch.nn import Module
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def plot_TSNE(encoder: Module, 
              dataset: Dataset,
              n_samples=200,
              perplexity=[20],
              batchsize=500,
              device='cuda',
              save_dir = None):

    encoder.to(device)
    encoder.eval()

    embedding_list = []
    feature_list = []

    feature = dataset.feature
    dataloader = dataset.get_dataloader_reg(batchsize=batchsize, TSNE = True, dataset_size = n_samples)

    size=0

    while size < n_samples:

        for inputs,targets in dataloader: 

            inputs = inputs.to(device)
            targets = targets

            # Save

            embedding_list.append(encoder(inputs).detach().cpu().numpy())
            feature_list.append(targets.numpy())

            size+=batchsize

            if size>n_samples:
                break

    
    embeddings=np.concatenate(embedding_list)
    features=np.concatenate(feature_list)

    for p in perplexity:
        print(f'Computing TSNE {p}')
        embeddings_proj = TSNE(n_components=2,perplexity=p).fit_transform(embeddings)


        if feature.type == 'continuous':
            
            if feature.dim>1:
                fig, axs = plt.subplots(feature.dim)
            else:
                fig, ax = plt.subplots()

            fig.suptitle(f'Perplexity: {p}, feature: {feature.name}')

            for k in range(feature.dim):
                
                if feature.dim>1:
                    ax = axs[k]

                attribute_list=[f[k] for f in features]

                cm = plt.cm.get_cmap('RdYlBu')
                scatter = ax.scatter(embeddings_proj[:,0],embeddings_proj[:,1],c=attribute_list, cmap=cm, alpha = 0.5)
                fig.colorbar(scatter)


        if feature.type == 'discrete':

            attribute_list=[f.item() for f in features]

            fig, ax = plt.subplots()
            fig.suptitle(f'Perplexity: {p}, feature: {feature.name}')

            cm = plt.cm.get_cmap('RdYlBu')
            scatter = ax.scatter(embeddings_proj[:,0],embeddings_proj[:,1],c=attribute_list, cmap=cm, alpha = 0.5)
            fig.colorbar(scatter)

                
            

        fig.savefig(f'plots/TNSE_{p}.pdf',bbox_inches='tight')
        if save_dir!=None:
            fig.savefig(save_dir+f'/TNSE_{p}.pdf',bbox_inches='tight')
        plt.show()