import user_define_modules
import argparse
from os import path


def parse_options(root_path):
    parser = argparse.ArgumentParser()
    default_config = path.join(root_path, 'config.yml')
    parser.add_argument('-opt', type=str, default=default_config, help='Config path.')


if __name__ == '__main__':
    root = path.abspath(path.join(__file__, path.pardir))
    parse_options(root)
