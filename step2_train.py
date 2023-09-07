from datasets.all_dataset import get_dataset
from step2.train import train
from models.utils import load_model,save_model,get_decoder,get_encoder
from torch.optim import Adam
from random import choice
import wandb

###### ----------------------------- PARSER ----------------------------- ######

import argparse
parser = argparse.ArgumentParser()

# General parameters of the run :

parser.add_argument('--device', default='cuda',
                    help='device')

parser.add_argument('--dataset', default='ZINC',
                    help='dataset')

parser.add_argument('--id', default=None,
                    help='name of the run (for saving model)')

parser.add_argument('--pretrained_encoder', default=1, type=int,
                    help='init encoder with the pretrained encoder (from metric learning)')

# Decoder architecture

parser.add_argument('--decoder', default='MLP',
                    help='decoder archi in {MLP,TRANSFORMER,2STEP}')

parser.add_argument('--random_search', default=0, type=int,
                    help='use random hypeparameters')

parser.add_argument('--depth', default=3, type=int,
                    help='number hidden of layer in decoder {1,3,5}')

parser.add_argument('--N_max', default='dilatation',
                    help='regime for N_max in {median,compression,dilatation}')

parser.add_argument('--hidden_dim', default=256, type=int,
                    help='common dimension of all hidden layer {64,128,256}')


# Training params

parser.add_argument('--batchsize', default=256, type=int,
                    help='batchsize')

parser.add_argument('--n_gradients', default=30000, type=int,
                    help='epochs = (batchsize*n_gradients)/len(dataset)')


args = parser.parse_args()

if args.random_search:
      args.depth = choice([1,2,3])
      args.N_max = choice(['median','compression','dilatation'])
      args.hidden_dim = choice([64,128,256])

print(vars(args))

###### ----------------------------- Init wandb ----------------------------- ######

if args.id == None:
      run = wandb.init(project="G2V2G_protocol",
                       tags=[args.decoder,args.dataset,'step2'],
                       config=vars(args))
      args.id = args.decoder + '_' + run.name
else:
      run = wandb.init(project="G2V2G_protocol",
                       tags=[args.decoder,args.dataset,'step2'],
                       name = args.id,
                       config=vars(args))
print(run.name)

###### ----------------------------- DATASET ----------------------------- ######

dataset = get_dataset(args.dataset)

if args.N_max == 'dilatation':
      N_max = dataset.size_qt095
if args.N_max == 'median':
      N_max = dataset.size_qt050
if args.N_max == 'compression':
      N_max = dataset.size_qt005

###### ----------------------------- ENCODER ----------------------------- ######

encoder = load_model('saves/step1/'+args.dataset+'/pretrained_encoder')

if not(args.pretrained_encoder):
      print('using a non pretrained encoder')
      encoder = get_encoder(encoder.params['name'],**encoder.params)

###### ----------------------------- DECODER ----------------------------- ######

decoder = get_decoder(decoder_name = args.decoder,
                      nodes_channels = dataset.node_dim,
                      embedding_channels = encoder.embedding_channels,
                      dropout = 0.,
                      N_max = N_max,
                      depth = args.depth,
                      edge_embedding_channels = args.hidden_dim//2,
                      nhead = args.hidden_dim//32,
                      d_transformer = args.hidden_dim,
                      d_ffn = 2*args.hidden_dim,     
                      )


###### ----------------------------- OPTIMIZER ----------------------------- ######

optimizer = Adam(params=[*encoder.parameters(),*decoder.parameters()], lr=1e-3)

###### ----------------------------- TRAIN ----------------------------- ######

save_dir = 'saves/step2/'+args.dataset+'/'+args.id

train(encoder = encoder,
      decoder = decoder,
      save_dir = save_dir,
      optimizer = optimizer,
      dataset = dataset,
      batchsize = args.batchsize,
      n_grad = args.n_gradients, 
      loss_fun_gw = 'square_loss',
      device = args.device
      )

###### ----------------------------- SAVE TRAINING ----------------------------- ######

save_model(encoder,dir = save_dir, model_name = 'encoder')
save_model(decoder,dir = save_dir, model_name = 'decoder')

