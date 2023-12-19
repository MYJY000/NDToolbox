# 用于运行离线的分析, 运行粒度为实验
# 在使用 build_exp 选取参数生成实验的 config.yml 后, 可以使用该脚本运行实验
# 例如假设新建了一个实验叫做: exp_sua_1, 那么只需在工程根目录下运行命令行
# python exp_run.py -e exp_sua_1
import argparse
import os
import sys
sys.path.append("../")
from ndbox.dataset import build_dataset
from ndbox.analyzer import run_analyze
from ndbox.utils import yaml_load, get_root_logger, opt2str

def parse_opt(opt: dict):
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiment', type=str, help='The experiment name.')
    args = parser.parse_args()
    experiments = args.experiment
    if experiments is None:
        experiments = os.listdir(opt['exp_root'])
    else:
        experiments = [experiments]
    opt['experiments'] = experiments

def analyze_file_handler(exp_name: str):
    import logging
    formatter = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s: %(message)s')
    file_handler = logging.FileHandler(exp_name)
    file_handler.setLevel(level=logging.INFO)
    file_handler.setFormatter(formatter)
    return file_handler

def run():
    # 1. 解析命令行参数
    opt = {
        'root': os.getcwd(),
        'exp_root': os.path.join(os.getcwd(), 'experiments'),
        'logger_name': 'analyzer'
    }
    parse_opt(opt)
    logger = get_root_logger(logger_name=opt['logger_name'])
    experiments = opt.pop('experiments')
    # 2. 运行实验
    for exp in experiments:
        opt['experiment'] = exp
        logger.info(f"Start experiment {exp}.")
        exp_root_path = os.path.join(opt['exp_root'], exp)
        if not os.path.exists(exp_root_path):
            logger.warning(f"There is no experiment named {exp}")
            continue

        exp_log_path = os.path.join(exp_root_path, 'log.log')
        exp_log_file_hd = analyze_file_handler(exp_log_path)
        logger.addHandler(exp_log_file_hd)
        logger.info("Logger prepared.")

        exp_config_path = os.path.join(exp_root_path, 'config.yml')
        if not os.path.exists(exp_config_path):
            logger.warning(f"No configuration file `config.yml` found in experiment {exp}")
            logger.removeHandler(exp_log_file_hd)
            continue
        exp_opt = yaml_load(exp_config_path)
        opt['exp_opt'] = exp_opt
        if exp_opt is None:
            logger.warning(f"Empty configuration in experiment {exp}")
            logger.removeHandler(exp_log_file_hd)
            continue
        logger.info(f"Load {exp}'s configuration success!")

        dataset_opt = exp_opt.get('dataset', None)
        if dataset_opt is None:
            logger.warning(f"No dataset configuration found in {exp}'s configuration")
            logger.removeHandler(exp_log_file_hd)
            continue
        dataset = build_dataset(dataset_opt)

        experiment_opt = {'experiment': exp_opt.get('experiment', None)}
        if experiment_opt['experiment'] is None:
            logger.warning(f"No experiment configuration found in {exp}'s configuration")
            logger.removeHandler(exp_log_file_hd)
            continue
        experiment_opt['root'] = exp_root_path
        experiment_opt['logger_name'] = opt['logger_name']
        logger.info("Prepare to run the experiment")
        logger.info(opt2str(experiment_opt))
        run_analyze(dataset, experiment_opt)
        logger.removeHandler(exp_log_file_hd)


if __name__ == '__main__':
    run()



