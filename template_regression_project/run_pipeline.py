import os
import random
import argparse
import time
from os import path
from collections import OrderedDict

from ndbox.utils import (yaml_load, opt2str, set_random_seed, get_root_logger,
                         load_image, restore_image)

try:
    import user_define_modules
except ImportError:
    pass


def parse_options():
    parser = argparse.ArgumentParser()
    default_config = path.join(root, 'config.yml')
    parser.add_argument('-opt', type=str, default=default_config, help='Config path.')
    args = parser.parse_args()
    opt = yaml_load(args.opt)

    opt['path'] = {}
    name = opt.get('name', 'result')
    temp_path = path.join(root, 'temp_files')
    result_path = path.join(root, 'results', f'{name}_{int(time.time())}')
    log_path = path.join(result_path, 'log.txt')

    opt['path']['root'] = root
    opt['path']['temp'] = temp_path
    opt['path']['result'] = result_path
    opt['path']['log'] = log_path
    os.makedirs(result_path)

    seed = opt.get('random_seed')
    if seed is None:
        seed = random.randint(1, 10000)
        opt['random_seed'] = seed
    set_random_seed(seed)

    return opt, args


def run_pipeline():
    opt, args = parse_options()
    logger = get_root_logger(log_file=opt['path']['log'])
    logger.info(opt2str(opt))
    result_path = str(opt['path']['result'])
    temp_path = opt['path']['temp']

    datasets_opt = opt.get('dataset', OrderedDict())
    experiments_opt = opt.get('experiment', OrderedDict())

    datasets = OrderedDict()
    for dataset_name, dataset_opt in datasets_opt.items():
        datasets[dataset_name] = load_image(temp_path, dataset_opt)

    for exp_name, exp_opt in opt.get('experiment', OrderedDict()).items():
        exp_path = path.join(result_path, exp_name)
        os.makedirs(exp_path)
        logger.info(f'Start experiment {exp_name}.')

        processor_opt = exp_opt.get('processor', OrderedDict())
        model_opt = exp_opt.get('model', OrderedDict())
        train_opt = exp_opt.get('train', OrderedDict())
        test_opt = exp_opt.get('test', OrderedDict())
        metrics_opt = exp_opt.get('metrics', OrderedDict())


if __name__ == '__main__':
    root = path.abspath(path.join(__file__, path.pardir))
    run_pipeline()
