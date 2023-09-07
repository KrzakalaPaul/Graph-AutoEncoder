

def get_dataset(dataset_name: str, load_distances = False):

    if dataset_name == 'SBM2_custom':
        from datasets.SBM2 import SBM2
        dataset = SBM2(normalize_w = True,
                       custom_metric = True,
                       load_distances = load_distances)

    if dataset_name == 'SBM2':
        from datasets.SBM2 import SBM2
        dataset = SBM2(normalize_w = True,
                       custom_metric = False,
                       alpha = 'auto',
                       load_distances = load_distances)
        
    if dataset_name == 'SBM1_custom':
        from datasets.SBM1 import SBM1
        dataset = SBM1(normalize_w = True,
                       custom_metric = True,
                       load_distances = load_distances)

    if dataset_name == 'SBM1':
        from datasets.SBM1 import SBM1
        dataset = SBM1(normalize_w = True,
                       custom_metric = False,
                       alpha = 'auto',
                       load_distances = load_distances)

    if dataset_name == 'ogbg_molhiv':
        from datasets.ogbg_molhiv import ogbg_molhiv
        dataset = ogbg_molhiv(normalize_w = True,
                              alpha = 'auto',
                              load_distances = load_distances)
        
    if dataset_name == 'ZINC':
        from datasets.ZINC import ZINC
        dataset = ZINC(normalize_w = True,
                       alpha = 'auto',
                       load_distances = load_distances)
    
    if dataset_name == 'TREES':
        from datasets.Tree import TREES
        dataset = TREES(normalize_w = True,
                       alpha = 'auto',
                       load_distances = load_distances)
        
    return dataset
