import os
import argparse
import time
from os import path

from ndbox.utils import create_directory_and_files, file2file


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('-name', type=str, default='-', help='Project name.')
    parser.add_argument('-mode', type=str, default='regression',
                        help="Project mode. Default 'regression'. "
                             "'r' is regression, "
                             "'c' is classification, "
                             "'a' is analyzer.")
    parser.add_argument('-path', type=str, default=root, help='Project path.')
    args = parser.parse_args()

    trans = {'r': 'regression', 'c': 'classification', 'a': 'analyzer'}
    args.mode = args.mode if trans.get(args.mode) is None else trans.get(args.mode)

    if args.name == '-':
        args.name = args.mode + '_project_' + str(int(time.time()))

    if args.mode == 'regression':
        create_regression_project(args)
    elif args.mode == 'classification':
        create_classification_project(args)
    elif args.mode == 'analyzer':
        create_analyzer_project(args)
    else:
        raise NotImplementedError(f'Mode {args.mode} not support!')


def create_regression_project(args):
    project_path, user_define_path = create_regression_project_structure(path.join(args.path, args.name))
    regression_pipeline_init(project_path)
    regression_web_ui_init(project_path)
    regression_result_report_init(project_path)


def create_classification_project(args):
    project_path, user_define_path = create_classification_project_structure(path.join(args.path, args.name))
    classification_pipeline_init(project_path)


def create_analyzer_project(args):
    # 1. create analyzer project structure
    project_path = path.join(args.path, args.name)
    if path.exists(project_path):
        raise FileExistsError(f"Project {project_path} already exists!")
    os.makedirs(project_path)
    dir_structure = {
        'experiments': {},      # each experiment will be carried here
        'build_exp': {
            'cache.yml': None,    # cache the basic info about the dataset
            'app.py': None,
            'create_exp.py': None,
            'dataset_loader.py': None,
            'execute.py': None,
            'history.py': None
        },
        'run_exp.py': None
    }
    analyzer_init_root = path.join('./project_init_files', 'analyzer', 'tmp')
    ana_path_list = analyzer_init_root.split(os.sep)
    analyzer_init(ana_path_list, project_path, dir_structure)


def analyzer_init(ana_path_list, root_path, dir_structure):
    for name, sub in dir_structure.items():
        sub_path = os.path.join(root_path, name)
        if sub is None:
            fln, flp = name.split('.')
            if flp != 'py':
                open(sub_path, 'w', encoding='utf-8').close()
            else:
                ana_path_list[-1] = fln
                src_path = os.sep.join(ana_path_list)
                file2file(src_path, sub_path)
        else:
            os.makedirs(sub_path)
            analyzer_init(ana_path_list, sub_path, sub)


def create_regression_project_structure(project_path):
    if path.exists(project_path):
        raise FileExistsError(f"Project {project_path} already exists!")
    os.makedirs(project_path)
    dir_structure = {
        'user_define_modules': {
            '__init__.py': None
        },
        'temp_files': {},
        'results': {},
        'config.yml': None,
        'web_ui.py': None,
        'result_report.py': None,
        'run_pipeline.py': None
    }
    create_directory_and_files(project_path, dir_structure)
    user_define_path = path.join(project_path, 'user_define_modules')
    user_define_init(path.join(user_define_path, '__init__.py'))
    return project_path, user_define_path


def user_define_init(filepath):
    src_filepath = path.join(root, 'project_init_files', 'user_define_init')
    file2file(src_filepath, filepath)


def regression_pipeline_init(project_path):
    src_filepath = path.join(root, 'project_init_files', 'regression', 'run_pipeline')
    filepath = path.join(project_path, 'run_pipeline.py')
    file2file(src_filepath, filepath)


def regression_web_ui_init(project_path):
    src_filepath = path.join(root, 'project_init_files', 'regression', 'web_ui')
    filepath = path.join(project_path, 'web_ui.py')
    file2file(src_filepath, filepath)


def regression_result_report_init(project_path):
    src_filepath = path.join(root, 'project_init_files', 'regression', 'result_report')
    filepath = path.join(project_path, 'result_report.py')
    file2file(src_filepath, filepath)


def create_classification_project_structure(project_path):
    if path.exists(project_path):
        raise FileExistsError(f"Project {project_path} already exists!")
    os.makedirs(project_path)
    dir_structure = {
        'user_define_modules': {
            '__init__.py': None
        },
        'temp_files': {},
        'results': {},
        'config.yml': None,
        'run_pipeline': None
    }
    create_directory_and_files(project_path, dir_structure)
    user_define_path = path.join(project_path, 'user_define_modules')
    user_define_init(path.join(user_define_path, '__init__.py'))
    return project_path, user_define_path


def classification_pipeline_init(project_path):
    src_filepath = path.join(root, 'project_init_files', 'classification', 'run_pipeline')
    filepath = path.join(project_path, 'run_pipeline.py')
    file2file(src_filepath, filepath)


if __name__ == '__main__':
    root = path.abspath(path.join(__file__, path.pardir))
    parse_options()
