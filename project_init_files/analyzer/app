import streamlit as st
from st_pages import Page, show_pages

app_pages = [
    Page("app.py", "Home", "🏠"),
    Page("dataset_loader.py", "Load Dataset", "🔴"),
    Page("create_exp.py", "Create Experiment", "🟠"),
    Page("execute.py", "Execute", "🟡"),
    Page("history.py", "Load history experiments", "🔵")
]
show_pages(app_pages)
st.title("A simple spike data processing helper -- NDToolbox")
st.header('')
st.text("Welcome to NDToolbox, enjoy your trip!")


def load_yml(fn: str):
    import yaml
    with open(fn, 'r', encoding='utf-8') as f:
        result = yaml.load(f.read(), Loader=yaml.FullLoader)
    return result

def dump_yml(fn: str, data: dict):
    import yaml
    with open(fn, 'w', encoding='utf-8') as f:
        yaml.dump(data=data, stream=f)

def cache_load():
    cache = load_yml('cache.yml')
    if cache is None:
        cache = {}
    return cache

def cache_dump(cache: dict):
    dump_yml('cache.yml', cache)

def exp_load():
    return load_yml('exp.yml')

def exp_dump(exp: dict):
    dump_yml('exp.yml', exp)

