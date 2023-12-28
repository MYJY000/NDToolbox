# NDBOX:

### Installation

Need python >= 3.10.
```buildoutcfg
python setup.py install
```

### Getting Start
- Create a regression project or use `template_regression_project`
```buildoutcfg
python create_project.py -mode r -name regression_project
```
- Use `web_ui.py` generate `config.yml` or edit `config.yml` directly
```buildoutcfg
cd regression_project
streamlit run web_ui.py
```
- Run 'run_pipeline.py' base on `config.yml`
```buildoutcfg
python run_pipeline.py
```
- Use `result_report.py` visualizing the results
```buildoutcfg
streamlit run result_report.py
```
