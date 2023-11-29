import os
import argparse
import time
from os import path

from ndbox.utils import create_directory_and_files


def parse_options(root_path):
    parser = argparse.ArgumentParser()
    parser.add_argument('-name', type=str, default='-', help='Project name.')
    parser.add_argument('-mode', type=str, default='r',
                        help="Project mode. Default 'r'. "
                             "'r' is regression, "
                             "'c' is classification, "
                             "'a' is analyzer.")
    parser.add_argument('-path', type=str, default=root_path, help='Project path.')
    args = parser.parse_args()

    if args.name == '-':
        args.name = args.mode + '_project_' + str(int(time.time()))

    if args.mode == 'r':
        create_regression_project(args)
    elif args.mode == 'c':
        create_classification_project(args)
    elif args.mode == 'a':
        create_analyzer_project(args)
    else:
        raise NotImplementedError


def create_regression_project(args):
    paths = create_project_structure(path.join(args.path, args.name))


def create_classification_project(args):
    pass


def create_analyzer_project(args):
    pass


def create_project_structure(project_path):
    if path.exists(project_path):
        raise FileExistsError(f"Project {project_path} already exists!")
    os.makedirs(project_path)
    dir_structure = {
        'user_define_modules': {
            '__init__.py': None
        },
        'config.yml': None,
        'config_generator.py': None,
        'config_generator_ui.py': None,
        'result_report.py': None,
        'run_pipeline.py': None
    }
    create_directory_and_files(project_path, dir_structure)
    result_path = path.join(project_path, 'result')
    user_define_path = path.join(project_path, 'user_define_modules')
    user_define_init(path.join(user_define_path, '__init__.py'))
    return [project_path, result_path, user_define_path]


def user_define_init(filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(
            """import glob
import importlib
from os import path


py_files = glob.glob(path.join(path.dirname(__file__), '*.py'))
module_names = [path.splitext(path.basename(py_file))[0] for py_file in py_files]
modules = [importlib.import_module(f'user_define_modules.{module_name}') for module_name in module_names]
"""
        )
        f.close()


if __name__ == '__main__':
    root = path.abspath(path.join(__file__, path.pardir))
    parse_options(root)
