from datasets.base import Dataset,Feature
import torch as torch
from torch_geometric.data import Batch
from torch.nn import CrossEntropyLoss,MSELoss,Module
from torch.optim import Adam
from torch_geometric.nn.models import MLP


def regression_eval(encoder: Module,
                    predictor: Module, 
                    dataset: Dataset,
                    batchsize = 512,
                    n_samples = 100,
                    device = 'cuda'
                    ):
    
    size = 0
    score = 0
    dataloader = dataset.get_dataloader_reg(batchsize=batchsize, n_samples = n_samples)
    feature = dataset.feature

    encoder.to(device)
    encoder.eval()

    predictor.to(device)
    predictor.eval()

    while size < n_samples:

        for inputs,targets in dataloader: 

            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = predictor(encoder(inputs))

            if feature.type == 'discrete':
                
                # Top 1 
                _, predicted_labels = torch.max(outputs, 1)
                _, target_labels = torch.max(targets, 1)
                score += (predicted_labels == target_labels).sum().item()
                '''
                _, predicted_labels = torch.topk(outputs, 2, dim=1)
                _, target_labels = torch.max(targets, 1)

                score += (predicted_labels[:,0] == target_labels).sum().item() + (predicted_labels[:,1] == target_labels).sum().item()
                '''
                
            if feature.type == 'continous':
                score += torch.sum((outputs-targets)**2).item()

            size+=batchsize

    # Normalization 
    if feature.type == 'discrete':
        score /= size

    if feature.type == 'continous':
        targets = torch.concat([feature(dataset.give_instance()).reshape(1,-1) for _ in range(batchsize)])
        avg = torch.mean(targets,dim=0)
        norm = torch.sum( (targets - avg)**2 ).item()
        score = torch.sqrt(score/norm)

    print(score)


def regression(encoder: Module, 
               dataset: Dataset,
               batchsize = 512,
               n_gradients = 100,
               device = 'cuda'
               ):
    
    feature = dataset.feature
    dataloader = dataset.get_dataloader_reg(batchsize=batchsize)

    # Define predictor

    predictor = MLP(in_channels=encoder.embedding_channels,
                    hidden_channels=feature.dim,
                    out_channels=feature.dim,
                    num_layers=3    
                    )
    
    encoder.to(device)
    encoder.train()

    predictor.to(device)
    predictor.train()

    # Define loss

    if feature.type == 'discrete':
        loss_fn = CrossEntropyLoss()
    elif feature.type == 'continuous':
        loss_fn = MSELoss()

    # Define optmizer

    optimizer = Adam(params=[*encoder.parameters(),*predictor.parameters()])

    # Run training

    g = 0

    while g < n_gradients:

        for inputs,targets in dataloader: 

            inputs = inputs.to(device)
            targets = targets.to(device)

            # Get pred 

            outputs = predictor(encoder(inputs))

            # Compute loss

            loss = loss_fn(outputs,targets)

            # Backprop

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            print(loss.item())

            g+=1

    return predictor
