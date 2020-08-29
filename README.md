ds_helper
==============================

Collection of helper functions for routine Data Science tasks. This is a compilation of functions aimed to speed up data wrangling and analysis, as well as help with data modelling tasks, such as:

```python
import pandas as pd
from df_helper import features

# import unbalanced datasets for a classification problem
x_train_unb =  pd.read_csv('/path/to/x_train.csv')
y_train_unb =  pd.read_csv('/path/to/y_train.csv')

# performing downsampling on the datasets so data can be fed
# into a classification model
x_train_bal, y_train_bal = features.downsampling(unb_x_train, unb_y_train)
```

And that's it, you have an improved training set for classification with just one function.


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


Project Organization
------------

    ├── LICENSE
    ├── README.md
    ├── requirements.txt
    └── ds_helper
        ├── data.py        <- Functions for importing/exporting data.
        ├── features.py    <- Functions for Data Wrangling and Feature Engineering.
        ├── model.py       <- Functions for Data Modelling.
        └── viz.py         <- Functions for visualizing data.

--------

License
-------
This project is licensed under the MIT license.

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
