import torch 
import torch_geometric
import wandb
from datasets.base import Dataset
from .loss import to_FCh_format,fgw_loss
from models.utils import save_model
from time import time
from .utils import alignement
from .eval import eval

def train(encoder: torch.nn.Module,
          decoder: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          dataset: Dataset,
          n_grad: int,
          batchsize: int,
          save_dir: str,
          loss_fun_gw = 'square_loss',
          device = 'cuda',
          ):
    
    encoder.to(device)
    decoder.to(device)

    encoder.train()
    decoder.train()

    alpha = dataset.alpha
    normalization = dataset.normalization
    loader_train = dataset.get_dataloader(batchsize=batchsize,dataset_size=100000)
    loader_eval = dataset.get_dataloader(batchsize=512,dataset_size=10000)

    g = 0

    t_total = 0
    t_fgw = 0 

    while g<n_grad:

        for inputs in loader_train:

            clock1 = time()

            batchsize = len(inputs)

            inputs = inputs.to(device)

            outputs_F, outputs_C, outputs_h = decoder(encoder(inputs))
            outputs = list(zip(outputs_F,outputs_C,outputs_h))

            targets_F, targets_C, targets_h = to_FCh_format(inputs, device = device )
            targets = list(zip(targets_F,targets_C,targets_h))

            loss_batch = 0
            grads_encoder = []
            grads_decoder = []

            clock2 = time()

            for output,target in zip(outputs,targets):
                loss_input = fgw_loss(output,target,alpha=alpha,loss_fun_gw=loss_fun_gw) 
                loss_batch += loss_input
                #grads_encoder.append(torch.autograd.grad(loss_input,encoder.get_weight(),retain_graph=True)[0].flatten())
                #grads_decoder.append(torch.autograd.grad(loss_input,decoder.get_weight(),retain_graph=True)[0].flatten())

            clock2 = time() - clock2

            loss = loss_batch/batchsize

            if normalization != None:
                loss /= normalization

            optimizer.zero_grad()
            loss.backward()

            clipping_value = 10 # arbitrary value of your choosing
            torch.nn.utils.clip_grad_norm(encoder.parameters(), clipping_value)
            torch.nn.utils.clip_grad_norm(decoder.parameters(), clipping_value)

            optimizer.step()
            
            print(f'Grad {g}/{n_grad}')
            print(loss.item())

            clock1 = time() - clock1

            t_total += clock1
            t_fgw += clock2 

            log_dic = {'gradients': g,
                       'loss': loss.item(), 
                       'total time ': t_total, 
                       'total fgw time ': t_fgw,
                       'step fgw time ': clock2
                        }
                       #'Grads Alignement (encoder)': alignement(grads_encoder),
                       #'Grads Alignement (decoder)': alignement(grads_decoder)}
            
            grads_encoder = encoder.get_grads()
            for k,grad in enumerate(grads_encoder):
                log_dic[f'Encoder layer {k} (grad inf-norm)'] = torch.abs(grad).max()

            grads_decoder = decoder.get_grads()
            for k,grad in enumerate(grads_decoder):
                log_dic[f'Decoder layer {k} (grad inf-norm)'] = torch.abs(grad).max()

            if g%1000==0:
                eval_fgw,_ = eval(encoder = encoder,
                                  decoder = decoder,
                                  loader_eval = loader_eval,
                                  alpha = alpha,
                                  normalization = normalization,
                                  loss_fun_gw = 'square_loss',
                                  device = 'cuda',
                                  save_dir = None
                                  )
                
                log_dic['eval_fgw'] = eval_fgw


            wandb.log(log_dic)

            g+=1

            if g>n_grad:
                break

        save_model(encoder,dir = save_dir, model_name = 'encoder')
        save_model(decoder,dir = save_dir, model_name = 'decoder')