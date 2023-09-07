import torch.nn as nn
import torch 
import pickle
import os
from models.encoder import Basic_Encoder,Gated_Encoder,Virtual_Encoder
from .decoder_mlp import DecoderMLP
from .decoder_2step import Decoder2STEP
from .decoder_transformer import DecoderTRANSFORMER


def get_encoder(encoder_name: str, **kwargs):

    if encoder_name == 'basic' or encoder_name == 'basic_encoder':
        return Basic_Encoder(**kwargs)

    if encoder_name == 'gated' or encoder_name == 'gated_encoder':
        return Gated_Encoder(**kwargs)
    
    if encoder_name == 'virtual' or encoder_name == 'virtual_encoder':
        return Virtual_Encoder(**kwargs)

def get_decoder(decoder_name: str, **kwargs):

    if decoder_name == 'MLP':
        return DecoderMLP(**kwargs)
    
    if decoder_name == 'TRANSFORMER':
        return DecoderTRANSFORMER(**kwargs)
    
    if decoder_name == '2STEP':
        return Decoder2STEP(**kwargs)

def save_model(model: nn.Module, dir: str, model_name = None):

    if not(os.path.exists(dir)):
        os.mkdir(dir)

    params = model.params

    if model_name == None:
        model_name = params['name']

    file = dir + '/' +model_name

    torch.save(model.state_dict(),file)

    with open(file + '_params.pickle', 'wb') as handle:
        pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_model(file: str):

    with open(file + '_params.pickle', 'rb') as handle:
        params = pickle.load(handle)

    name = params['name']

    if name == 'basic_encoder':
        model = Basic_Encoder(**params)

    if name == 'gated_encoder':
        model = Gated_Encoder(**params)

    if name == 'virtual_encoder':
        model = Virtual_Encoder(**params)

    if name =='TRANSFORMER_decoder':
        model = DecoderTRANSFORMER(**params)

    if name =='MLP_decoder':
        model = DecoderMLP(**params)

    print(file)
    print(params)
    model.load_state_dict(torch.load(file))

    return model


