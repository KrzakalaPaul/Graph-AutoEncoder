# protect the entry point
if __name__ == '__main__':
    
    from os import cpu_count
    print(cpu_count())

    from datasets.ogbg_molhiv import ogbg_molhiv as Dataset
    name = 'ogbg_molhiv'

    #from datasets.Tree import TREES as Dataset
    #name = 'TREES'

    #from datasets.SBM1 import SBM1 as Dataset
    #name = 'SBM1'

    #from datasets.SBM2 import SBM2 as Dataset
    #name = 'SBM2'

    #from datasets.ZINC import ZINC as Dataset
    #name = 'ZINC'

    from datasets.base import RealDataset,SyntheticDataset

    print('importation done!')

    ### ----------------------- If needed : precompute some instance  ----------------------- ###
    '''
    n_precomputed_instances = 100000
    import pickle

    if issubclass(Dataset,SyntheticDataset):

        precomputed_instances = [Dataset().get_instance().to_torch_data() for _ in range(n_precomputed_instances)]

        with open('saves/'+name+f'_Data/precomputed_instances.pickle', 'wb') as handle:
            pickle.dump(precomputed_instances, handle)
    '''
    ### ----------------------- Compute some stats ----------------------- ###

'''
    import numpy as np
    import pickle 

    print('Computing the stats...')
    if issubclass(Dataset,RealDataset):
        data = Dataset(normalize_w = False).data[:10000]
    elif issubclass(Dataset,SyntheticDataset):
        data = Dataset(normalize_w = False).load_precomputed_instances()[:10000]

    sizes = np.array([len(G.x) for G in data])

    print('Quantile 5%')
    print(np.quantile(sizes,0.05))
    print('Quantile 95%')
    print(np.quantile(sizes,0.95))
    print('Quantile 50 (median)%')
    print(np.quantile(sizes,0.5))
'''
   

    ### ----------------------- Compute Unnormalized GW and W ----------------------- ###
'''
import numpy as np
from ot.gromov import gromov_wasserstein2,fused_gromov_wasserstein2
from ot import emd2
from ot.utils import unif,dist
import torch
from torch_geometric.utils import to_dense_adj
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from multiprocessing.pool import Pool

def task(g1,g2):

    M = dist(g1.x,g2.x)
    p1 = torch.tensor(unif(len(g1.x)))
    p2 = torch.tensor(unif(len(g2.x)))
    C1 = to_dense_adj(edge_index = g1.edge_index,max_num_nodes =len(g1.x)).squeeze()
    C2 = to_dense_adj(edge_index = g2.edge_index,max_num_nodes =len(g2.x)).squeeze()

    w = emd2(a = p1, b = p2, M = M).item()
    gw = gromov_wasserstein2(C1=C1,C2=C2,p=p1,q=p2).item()
    
    return w,gw

if __name__ == '__main__':


    import numpy as np
    from ot.gromov import gromov_wasserstein2
    from ot import emd2
    from ot.utils import unif,dist
    import torch 
    from torch_geometric.utils import to_dense_adj
    import pandas as pd

    n_fgw = 10000

    if issubclass(Dataset,RealDataset):
        data = Dataset(normalize_w = False).data
    elif issubclass(Dataset,SyntheticDataset):
        data = Dataset(normalize_w = False).load_precomputed_instances()

    dic = {'i': [], 'j': [], 'w' : [], 'gw' : []}

    I = np.random.randint(0,len(data),n_fgw)
    J = np.random.randint(0,len(data),n_fgw)

    G1 = [data[i] for i in I]
    G2 = [data[j] for j in J]

    # create and configure the process pool
    with Pool() as pool:
        # execute tasks in order
        computations = pool.starmap(task, zip(G1,G2))
        
        for i,j,c in zip(I,J,computations):

            w,gw,fgw = c
    
            dic['i'].append(i)
            dic['j'].append(j)
            dic['gw'].append(gw)
            dic['w'].append(w)

    dataframe = pd.DataFrame(dic)
    dataframe.to_csv('saves/'+name+'_Data/unnormalized_gw.csv')

    print('avg w')
    print(dataframe['w'].mean())
    print('avg gw')
    print(dataframe['gw'].mean())
    print('sigma normalization')
    print(dataframe['w'].mean()/dataframe['gw'].mean())

    ### ----------------------- Compute all fGW (with normalization) ----------------------- ###


import numpy as np
from ot.gromov import gromov_wasserstein2,fused_gromov_wasserstein2
from ot import emd2
from ot.utils import unif,dist
import torch
from torch_geometric.utils import to_dense_adj
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from multiprocessing.pool import Pool


n_alphas = 10 # 2xn_alphas + 3

dic_alphas = {}
for k in range(1,n_alphas):
    dic_alphas[k] = k/n_alphas

def task(g1,g2):

    M = dist(g1.x,g2.x)
    p1 = torch.tensor(unif(len(g1.x)))
    p2 = torch.tensor(unif(len(g2.x)))
    C1 = to_dense_adj(edge_index = g1.edge_index,max_num_nodes =len(g1.x)).squeeze()
    C2 = to_dense_adj(edge_index = g2.edge_index,max_num_nodes =len(g2.x)).squeeze()

    w = emd2(a = p1, b = p2, M = M).item()
    gw = gromov_wasserstein2(C1=C1,C2=C2,p=p1,q=p2).item()
    
    fgw = []

    for k in dic_alphas.keys():
        fgw.append(fused_gromov_wasserstein2(M,C1=C1,C2=C2,p=p1,q=p2,alpha=dic_alphas[k]).item())
    
    return w,gw,fgw

if __name__ == '__main__':

    from time import time

    n_fgw = 100000

    with open('saves/'+name+'_Data/dic_alphas.pickle', 'wb') as handle:
        pickle.dump(dic_alphas, handle, protocol=pickle.HIGHEST_PROTOCOL)

    dic = {'i': [], 'j': [], 'w' : []}
    for k in dic_alphas.keys():
        dic[k] = []
    dic['gw'] = []

    print(dic)
    print(dic_alphas)

    if issubclass(Dataset,RealDataset):
        data = Dataset(normalize_w = True).data
    elif issubclass(Dataset,SyntheticDataset):
        data = Dataset(normalize_w = True).load_precomputed_instances()

    I = np.random.randint(0,len(data),n_fgw)
    J = np.random.randint(0,len(data),n_fgw)

    G1 = [data[i] for i in I]
    G2 = [data[j] for j in J]

    t = time()
    # create and configure the process pool
    with Pool() as pool:
        # execute tasks in order
        computations = pool.starmap(task, zip(G1,G2))
        
        for i,j,c in zip(I,J,computations):

            w,gw,fgw = c
    
            dic['i'].append(i)
            dic['j'].append(j)
            dic['gw'].append(gw)
            dic['w'].append(w) 
            for l,k in enumerate(list(dic_alphas.keys())):
                dic[k].append(fgw[l])
    
    print(time()-t)

    dataframe = pd.DataFrame(dic)
    dataframe.to_csv('saves/'+name+'_Data/all_fgw.csv')

'''
    ### ----------------------- Compute alpha star ----------------------- ###

if __name__ == '__main__':

    import pandas as pd
    import numpy as np
    import pickle

    dataframe = pd.read_csv('saves/'+name+'_Data/all_fgw.csv')
    with open('saves/'+name+'_Data/dic_alphas.pickle', 'rb') as handle:
        dic_alphas = pickle.load(handle)

    x = np.mean(np.array(dataframe.iloc[:,3:]),axis=0)
    alphas = np.array([0]+[dic_alphas[k] for k in dic_alphas.keys()] + [1])

    k_star = np.argmax(x)
    alpha_star = alphas[k_star]

    print('alpha start:')
    print(alpha_star)



    dataframe = dataframe[['i','j',str(k_star)]]
    dataframe = dataframe.rename(columns={str(k_star):'dist'})

    dataframe.to_csv('saves/'+name+'_Data/default_fgw.csv')

    print('Normalization:')
    normalization = dataframe['dist'].mean()
    print(normalization)

    ### ----------------------- Plot alpha star ----------------------- ###

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import pickle
    
    dataframe = pd.read_csv('saves/'+name+'_Data/all_fgw.csv')
    with open('saves/'+name+'_Data/dic_alphas.pickle', 'rb') as handle:
        dic_alphas = pickle.load(handle)

    #print(dataframe.head(10))

    x = np.mean(np.array(dataframe.iloc[:,3:]),axis=0)
    alphas = np.array([0]+[dic_alphas[k] for k in dic_alphas.keys()] + [1])

    fig, ax = plt.subplots()
    

    ax.text(alpha_star, normalization - (normalization-x[0])/10, r'$\alpha^*$' )
    ax.plot(alpha_star, normalization, 'ro')

    plt.plot(alphas,x) 
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$f(\alpha)$')
    plt.title(f'Dataset: {name},'+r' $\alpha^*$='+f'{alpha_star}', fontsize=15)
    plt.tight_layout()
    fig.savefig(f'plots/test.pdf')
    plt.show()



