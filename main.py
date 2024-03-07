import sys
import os.path as osp
import yaml
from auto_gk import autoGraphKernel, combination_result


if __name__ == '__main__':
    
    dataSet = sys.argv[1]
    number = sys.argv[2]
    test_path =  f"./experiment/{dataSet}/{number}"
    print("Start with", test_path)

    with open(osp.join(test_path, 'para.yml'), 'r') as f:
        kwargs = yaml.safe_load(f)
        kwargs['save_root'] = test_path

    autoGraphKernel(kwargs)
    # combination_result(kwargs)
    print('Finished!')