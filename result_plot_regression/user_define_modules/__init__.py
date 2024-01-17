import glob
import importlib
from os import path


py_files = glob.glob(path.join(path.dirname(__file__), '*.py'))
module_names = [path.splitext(path.basename(py_file))[0] for py_file in py_files]
modules = [importlib.import_module(f'user_define_modules.{module_name}') for module_name in module_names]
