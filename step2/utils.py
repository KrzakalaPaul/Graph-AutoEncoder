import torch

def alignement(grads_list):

    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    # grads_list is list of gradients of len(Batchsize)
    grads_list = torch.vstack(grads_list)
    normalized_grads = grads_list/(torch.linalg.norm(grads_list,dim=1,keepdim=True)+1e-8)
    aligment = torch.linalg.norm(torch.mean(normalized_grads,dim=0))

    return aligment
    