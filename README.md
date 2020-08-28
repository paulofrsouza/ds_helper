ds_helper
==============================

Collection of helper functions for routine Data Science tasks.


Installation
------------

```sh
# Download the repository from GitHub
git clone https://github.com/paulofrsouza/ds_helper.git                     
cd ds_helper
# Create empty virtual environment
virtualenv -p /usr/bin/python3.6 ds_helper_env
source ds_helper_env/bin/activate
# Install packages listed in requirements.txt
pip3 install -r requirements.txt
# Install the ds_helper package in development mode
pip3 install -e .
```

Utilization
-----------

```python
from ds_helper import data, features, model, viz
```

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    └── ds_helper          <- Source code used in the project.
        ├── __init__.py    <- Makes ds_helper a Python module
        ├── data.py        <- Functions for importing/exporting data.
        ├── features.py    <- Functions for Data Wrangling and Feature Engineering.
        ├── model.py       <- Functions for Data Modelling.
        └── viz.py         <- Functions for visualizing data.

--------

License
-------
MIT

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
