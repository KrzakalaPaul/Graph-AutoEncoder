from datasets.all_dataset import get_dataset
from models.utils import save_model,get_encoder
from step1.metric_learning import metric_learning
from step1.regression import regression
import wandb

###### ----------------------------- PARSER ----------------------------- ######

import argparse
parser = argparse.ArgumentParser()

# General parameters of the run :

parser.add_argument('--device', default='cuda',
                    help='device')

parser.add_argument('--dataset', default='ZINC')

parser.add_argument('--id', default=None,
                    help='name of the run (for saving model)')

# Encoder architecture

parser.add_argument('--encoder', default='gated',
                    help='decoder archi in {basic,gated,virtual}')

parser.add_argument('--depth', default=3, type=int,
                    help='number hidden of layer in encoder')

parser.add_argument('--embedding_dim', default=256, type=int,
                    help='common dimension of all hidden layer')

# Training params

parser.add_argument('--batchsize', default=512, type=int,
                    help='batchsize')

parser.add_argument('--n_gradients', default=10, type=int,
                    help='epochs = (batchsize*n_gradients)/len(dataset)')


args = parser.parse_args()

###### ----------------------------- Init wandb ----------------------------- ######

if args.id == None:
      run = wandb.init(project="G2V2G_protocol",
                       tags=[args.encoder,args.dataset,'step1'],
                       config=vars(args))
      args.id = args.encoder + '_' + run.name
else:
      run = wandb.init(project="G2V2G_protocol",
                       tags=[args.encoder,args.dataset,'step1'],
                       name = args.id,
                       config=vars(args))
      
###### ----------------------------- DATASET ----------------------------- ######

dataset = get_dataset(args.dataset, load_distances = True)

###### ----------------------------- ENCODER ----------------------------- ######

encoder = get_encoder(encoder_name = args.encoder,
                      in_channels = dataset.node_dim,
                      hidden_channels = args.embedding_dim,
                      embedding_channels = args.embedding_dim,
                      depth = args.depth
                      )

###### ----------------------------- METRIC LEARNING ----------------------------- ######

metric_learning(encoder=encoder,
                dataset=dataset,
                device=args.device,
                batchsize=args.batchsize,
                n_gradients=args.n_gradients     
                )

###### ----------------------------- SAVE TRAINING ----------------------------- ######

save_dir = 'saves/step1/'+args.dataset+'/'+args.id
save_model(encoder,dir = save_dir, model_name = 'encoder')





''' 
OLD :

#### ---------------- regression evaluation ---------------- ####

predictor = regression(encoder=encoder,
                       dataset=dataset,
                       device=args.device,
                       n_gradients=200)


torch.save(encoder.state_dict(), savefile+'/regression/encoder_'+encoder_name)
torch.save(predictor.state_dict(), savefile+'/regression/predictor_'+encoder_name)
'''