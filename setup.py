from setuptools import setup, find_packages

setup(
    name='ndbox',
    packages=find_packages(),
    python_requires='>=3.10',
    install_requires=[
        'joblib~=1.3.1',
        'numpy~=1.25.2',
        'scikit-learn~=1.3.0',
        'tqdm~=4.65.1',
        'pyyaml~=6.0.1',
        'pandas~=1.3.4',
        'pynwb~=2.4.0',
        'scipy~=1.11.1',
        'setuptools~=58.0.4',
        'streamlit~=1.29.0'
    ]
)
