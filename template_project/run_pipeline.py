import os
import random
import argparse
import time
from os import path
from copy import deepcopy

from ndbox.dataset import build_dataset
from ndbox.processor import run_processor
from ndbox.utils import yaml_load, opt2str, set_random_seed, get_root_logger

try:
    import user_define_modules
except ImportError:
    pass


def parse_options(root_path):
    parser = argparse.ArgumentParser()
    default_config = path.join(root_path, 'config.yml')
    parser.add_argument('-opt', type=str, default=default_config, help='Config path.')
    args = parser.parse_args()
    opt = yaml_load(args.opt)

    opt['path'] = {}
    name = opt.get('name', 'result')
    result_path = path.join(root_path, f'{name}_{int(time.time())}')
    log_path = path.join(result_path, 'log.txt')

    opt['path']['root'] = root_path
    opt['path']['result'] = result_path
    opt['path']['log'] = log_path
    os.makedirs(result_path)

    seed = opt.get('random_seed')
    if seed is None:
        seed = random.randint(1, 10000)
        opt['random_seed'] = seed
    set_random_seed(seed)

    return opt, args


def run_pipeline(root_path):
    opt, args = parse_options(root_path)
    logger = get_root_logger(log_file=opt['path']['log'])
    logger.info(opt2str(opt))

    dataset = {}
    for dataset_name, dataset_opt in opt.get('dataset', {}).items():
        dataset[dataset_name] = build_dataset(dataset_opt)

    result_path = opt['path']['result']
    for exp_name, exp_opt in opt.get('experiment', {}).items():
        exp_path = path.join(result_path, exp_name)
        os.makedirs(exp_path)
        p_dataset = processor_pipeline(dataset, exp_opt.get('processor', {}))
        model_opt = exp_opt.get('model')
        if model_opt is not None:
            raise NotImplementedError


def processor_pipeline(dataset, processor_opt):
    p_dataset = {}
    for dataset_name, dataset_opt in processor_opt.items():
        is_copy = dataset_opt.get('is_copy', True)
        if is_copy:
            nwb_dataset = deepcopy(dataset[dataset_name])
        else:
            nwb_dataset = dataset[dataset_name]
        for p_name, p_opt in dataset_opt.items():
            if isinstance(p_opt, dict):
                run_processor(nwb_dataset, p_opt)
        p_dataset[dataset_name] = nwb_dataset

    return p_dataset


if __name__ == '__main__':
    root = path.abspath(path.join(__file__, path.pardir))
    run_pipeline(root)
