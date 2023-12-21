import os
from glob import glob


def files_form_folder(folder, filename='*'):
    """
    Return a list of paths matching a pathname pattern.

    The pattern may contain simple shell-style wildcards à la
    fnmatch. However, unlike fnmatch, filenames starting with a
    dot are special cases that are not matched by '*' and '?'
    patterns.
    """

    if not os.path.exists(folder):
        raise FileNotFoundError(f"'{folder}' Directory not found!")
    if not os.path.isdir(folder):
        raise NotADirectoryError(f"The path '{folder}' is not a directory!")
    filenames = sorted(glob(os.path.join(folder, filename)))
    return filenames


def create_directory_and_files(root_path, dir_structure):
    for name, sub in dir_structure.items():
        sub_path = os.path.join(root_path, name)
        if sub is None:
            open(sub_path, 'w', encoding='utf-8').close()
        else:
            os.makedirs(sub_path)
            create_directory_and_files(sub_path, sub)


def file2file(src_path, dest_path):
    with open(src_path, 'r', encoding='utf-8') as f:
        src_content = f.read()
    with open(dest_path, 'w', encoding='utf-8') as f:
        f.write(src_content)
