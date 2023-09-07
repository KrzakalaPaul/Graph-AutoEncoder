from datasets.all_dataset import get_dataset
from step2.eval import eval
from models.utils import load_model
from step1.TNSE import plot_TSNE
from step1.embedding_sklearn import eval_embedding
from step2.recons import plot_recons

###### ----------------------------- PARSER ----------------------------- ######

import argparse
parser = argparse.ArgumentParser()

# General parameters of the run :

parser.add_argument('--device', default='cuda',
                    help='device')

parser.add_argument('--dataset', default='TREES',
                    help='dataset')

parser.add_argument('--id', default='MLP_drawn-wind-224',
                    help='id of the autoencoder to eval')

args = parser.parse_args()

###### ----------------------------- DATASET ----------------------------- ######

dataset = get_dataset(args.dataset,load_distances = True)

###### ----------------------------- ENCODER/DECODER ----------------------------- ######

save_dir = 'saves/step2/'+args.dataset+'/'+args.id
encoder = load_model(save_dir+'/encoder')
decoder = load_model(save_dir+'/decoder')


###### ----------------------------- EVAL ----------------------------- ######

loss_fgw,loss_w = eval(encoder = encoder,
                       decoder = decoder,
                       dataset = dataset,
                       n_samples = 3000,
                       batchsize = 512,
                       loss_fun_gw = 'square_loss',
                       device = args.device,
                       save_dir = save_dir
                       )

###### ---------------------------- EMBEDDING EVAL (TSNE) ----------------------------- ######


plot_TSNE(encoder=encoder,
          dataset=dataset,
          n_samples=2000,
          perplexity=[2,20,50],
          batchsize=512,
          device=args.device,
          save_dir = save_dir
          )


###### ----------------------------- EMBEDDING EVAL (REGRESSION) ----------------------------- ######

eval_embedding(encoder = encoder, 
               dataset = dataset,
               n_samples = 100000,
               batchsize = 512,
               device = args.device,
               save_dir = save_dir)

###### ----------------------------- RECONSTRUCTION PLOT ----------------------------- ######

plot_recons(encoder = encoder,
            decoder = decoder,
            dataset = dataset,
            n_recons = 10,
            device = args.device,
            save_dir = save_dir)
