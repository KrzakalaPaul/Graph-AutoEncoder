from datasets.all_dataset import get_dataset
from models.utils import load_model
from step1.metric_learning import metric_learning_eval
from step1.regression import regression_eval
from step1.TNSE import plot_TSNE
from step1.embedding_sklearn import eval_embedding

###### ----------------------------- PARSER ----------------------------- ######

import argparse
parser = argparse.ArgumentParser()

# General parameters of the run :

parser.add_argument('--device', default='cuda',
                    help='device')

parser.add_argument('--dataset', default='ogbg_molhiv',
                    help='dataset')

parser.add_argument('--id', default='big_virtual',  #gated_leafy-voice-232
                    help='name of the encoder to eval')

args = parser.parse_args()


###### ----------------------------- DATASET ----------------------------- ######
 
dataset = get_dataset(args.dataset, load_distances=True)

###### ----------------------------- ENCODER ----------------------------- ######

save_dir = 'saves/step1/'+args.dataset+'/'+args.id
save_file = save_dir+'/encoder'
encoder = load_model(save_file)

###### ----------------------------- ML EVAL ----------------------------- ######

metric_learning_eval(encoder=encoder,
                     dataset=dataset,
                     batchsize = 512,
                     device=args.device,
                     n_samples = 1000,
                     save_dir = save_dir
                     )

###### ----------------------------- EMBEDDING EVAL (TSNE) ----------------------------- ######
'''
plot_TSNE(encoder=encoder,
          dataset=dataset,
          n_samples=2000,
          perplexity=[2,20,50],
          batchsize=512,
          device=args.device,
          save_dir = save_dir
          )
'''
###### ----------------------------- EMBEDDING EVAL (REGRESSION) ----------------------------- ######
'''
eval_embedding(encoder = encoder, 
               dataset = dataset,
               n_samples = 100000,
               batchsize = 512,
               device = args.device,
               save_dir = save_dir
               )
'''