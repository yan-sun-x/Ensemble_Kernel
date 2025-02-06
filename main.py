from utils.arguments import arg_parse
import os
import os.path as osp
import yaml
import json
import numpy as np
from models import UMKL, UMKL_G, sparse_UMKL
from utils.get_metric import individual_results


if __name__ == '__main__':
    
    args = arg_parse()
    test_path = f"./experiment/{args.dataset}"
    print("Start with", test_path)

    with open(osp.join(test_path, 'para.yml'), 'r') as f:
        kwargs = yaml.safe_load(f)
    kernelNameList = kwargs['kernel_name_list']

    saved_folder = osp.join(test_path, str(args.num))
    if not osp.exists(saved_folder):
        os.mkdir(saved_folder)

    forward_suffix = ''
    if args.forward_loss and (args.loss_fun in ['PKL', 'MKL']): forward_suffix = '_forward'

    if args.model == 'UMKL-G': # our method
        base_name = f'results_{args.model}_{args.loss_fun}'
        if args.loss_fun in ['PKL', 'PCE']:
            base_name += f'_{args.power}'
            if args.set_fixed_ground_truth:
                base_name += '_fixed_Q'
        if args.init_type in ['eigen', 'eigen_inv', 'random']:
            base_name += f'_init_{args.init_type}'
        if args.perturb:
            base_name += f'_perturb_{args.noise_std}'
        if args.test:
            base_name += '_test'    
        
        base_name += '_time' #TODO: remove this line if you don't want to record the time
        
        saved_file = osp.join(saved_folder, f'{base_name}{forward_suffix}.json')
        saved_file_p = osp.join(saved_folder, f'p_{base_name}{forward_suffix}.npy')
        if osp.exists(saved_file): raise ValueError(f'{saved_file} Already Trained!')
        
        if args.test:
            print('Test mode')
            results, p_list = UMKL_G.autoGraphKernelTest(args, kernelNameList)
        else:
            print('Train mode')
            results, p_list = UMKL_G.autoGraphKernel(args, kernelNameList)
        # np.save(saved_file_p, np.array(p_list, dtype=object), allow_pickle=True)

    elif args.model == 'UMKL': # baseline 1
        base_name = f'results_baseline_{args.model}'
        base_name += '_time'
        saved_file = osp.join(saved_folder, f'{base_name}.json')
        if osp.exists(saved_file): raise ValueError('Already Trained!')
        results = UMKL.autoGraphKernel(args, kernelNameList)

    elif args.model == 'sparse-UMKL': # baseline 2
        base_name = f'results_baseline_{args.model}_{args.n_neighbors}'
        base_name += '_time'
        saved_file = osp.join(saved_folder, f'{base_name}.json')
        if osp.exists(saved_file): raise ValueError('Already Trained!')
        results = sparse_UMKL.autoGraphKernel(args, kernelNameList)
    
    elif args.model == 'individual': # baselines
        saved_file = osp.join(saved_folder, f'results_baseline_{args.model}.json')
        if osp.exists(saved_file): raise ValueError('Already Evaluated!')
        results = individual_results(args, kernelNameList)
    
    else:
        raise ValueError('Invalid model')
    
    with open(saved_file, 'w') as f:
        json.dump(results, f, indent=4)
    print('Results are saved in %s'%(saved_file))

    if not osp.exists(osp.join(saved_folder, 'para.yml')):
        kwargs['epochs'] = args.epochs
        kwargs['init_type'] = args.init_type
        kwargs['lr'] = args.lr
        kwargs['weight_decay'] = args.weight_decay
        with open(osp.join(saved_folder, 'para.yml'), 'w') as f:
            yaml.dump(kwargs, f, default_flow_style=False)
        print('Configurations are saved in %s'%(osp.join(saved_folder, 'para.yml')))
    
    print('Finished!')