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

The package's full documentation can be found at https://paulofrsouza.github.io/ds_helper/


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


Dependencies
------------

The package relies heavily on [Pandas](https://pandas.pydata.org/pandas-docs/stable/index.html), expanding and building over many of its functions. Given its capabilities for data manipulation, it is highly recommended to have a good understanding of it. The [10 minutes to Pandas](https://pandas.pydata.org/pandas-docs/stable/user_guide/10min.html) tutorial is a great starting point.

For data modelling and advanced data transformations, the package leverages [scikit-learn](https://scikit-learn.org/stable/index.html), given its broad industry application and reliability. [Scikit's documentation](https://scikit-learn.org/stable/getting_started.html) is a Data Science course on its own and a worth read.

It was used [matplotlib](https://matplotlib.org/users/index.html) for data visualization. Although its steep learning curve, it's a very powerful tool. [Seaborn](https://seaborn.pydata.org/tutorial.html) is also used for data viz and presents a more user-friendly interface.

--------

License
-------
This project is licensed under the MIT license.

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
