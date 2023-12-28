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
    project_path, user_define_path = create_project_structure(path.join(args.path, args.name))
    regression_pipeline_init(project_path)
    regression_web_ui_init(project_path)
    regression_result_report_init(project_path)


def create_classification_project(args):
    pass


def create_analyzer_project(args):
    # 1. create analyzer project structure
    project_path = path.join(args.path, args.name)
    if path.exists(project_path):
        raise FileExistsError(f"Project {project_path} already exists!")
    os.makedirs(project_path)
    dir_structure = {
        'experiments': {},      # each experiment will be carried here
        'analyze_info': {
            'info.yml': None    # cache the basic info about the dataset
        },
        'main_ui.py': None,
        'main.py': None
    }
    create_directory_and_files(project_path, dir_structure)

    from ndbox import analyzer
    ana_path = os.path.abspath(analyzer.__file__)
    ana_path_list = ana_path.split(os.sep)

    # 2. initialize the offline pipeline
    pipeline_path = path.join(project_path, 'run_exp.py')
    ana_path_list[-1] = analyzer.pipeline_path
    analyzer_init(ana_path_list, pipeline_path)
    # 3. initialize the ui pipeline
    pipeline_ui_path = path.join(project_path, 'build_exp.py')
    ana_path_list[-1] = analyzer.pipeline_ui_path
    analyzer_init(ana_path_list, pipeline_ui_path)


def analyzer_init(src, desc):
    src_path = os.sep.join(src)
    with open(src_path, 'r') as sf:
        src_content = sf.read()
    with open(desc, 'w') as df:
        df.write(src_content)


def create_project_structure(project_path):
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


if __name__ == '__main__':
    root = path.abspath(path.join(__file__, path.pardir))
    parse_options()
