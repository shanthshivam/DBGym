
DBGym Quick Start
========================

0 Your first DBGym experiment
-----------------------------

With less than 5 lines of code, you can set up and carry out experiments using DBGym.

.. code-block:: bash

  from dbgym.run import run
  from dbgym.config import get_config

  # Get default DBGym config
  config = get_config()
  stats = run(config)


The experiment statistics and predictions returned from the `run()` function and are further saved in the `output` directory by default.

1 Run DBGym experiments on RDBench datasets
-------------------------------------------

For users who want to customize DBGym experiments, we provide two ways to customize configuration: using Python and/or using `YAML` config file. Please refer to [`dbgym/config.py`](dbgym/config.py) for all the available configurations.

**Customize configs with Python**


You can easily set customized config values in the Python code, e.g., picking an RDBench dataset and writing your favorite predictive query. For example:

.. code-block:: bash
  
  from dbgym.run import run
  from dbgym.config import get_config

  config = get_config()
  # point to an RDBench dataset. Will auto-download if not cached
  config.dataset.name = 'rdb1-ather'
  # predictive query. Format: table_name.column_name
  config.dataset.query = 'entry_examination.Cholesterol'
  stats = run(config)


**Customize configs with YAML file**

Alternatively, you can write a customized config in a YAML file, e.g., `config_gcn.yaml`, and load with DBGym:

.. code-block:: bash

  from dbgym.run import run
  from dbgym.config import get_config

  config = get_config()
  config.merge_from_file('config_gcn.yaml')
  stats = run(config)


where the yaml `config_gcn.yaml` can be defined as follows:

.. code-block:: bash

  # Example Experiment Configuration for GCN
  train:
    epoch: 200

  model:
    name: 'GCN'
    hidden_dim: 128
    layer: 3

  optim:
    lr: 0.001


2 Apply DBGym to your own database datasets
-------------------------------------------

You can also easily apply DBGym to your own database datasets. To start, you can first organize your customized data as follows:

.. code-block:: bash

  dataset_path
  ├── dataset_1
  │   ├── user.csv
  │   ├── item.csv
  │   └── ...
  ├── dataset_2
  └── ...


Within each CSV file, DBGym assumes that most of the table columns can be arbitrarily named, except for the key columns, where you should *follow the naming convention*, where the primary key column for table `table` is named as `_table`. For example, the column names for each CSV file may look like:

.. code-block:: bash

  user.csv:	_user, x1, x2, ...
  item.csv:	_item, Feature1, Feature2, ...

where the special columns begin with `_` indicate key columns, and the remaining columns (can be arbitrarily named) are regarded as feature columns. For example, `_user` is the primary key column for `user.csv`, which should save the unique ID for each user; other tables could also refer to user ID information, e.g., by adding `_user` column in `trans.csv`, in which case `_user` is the foreign key column in `trans.csv` that could have non-unique values.


Then, you can point DBGym to the dataset directory, specify the dataset you want, and write a simple predictive query by pointing to any feature column:

.. code-block:: bash

  from dbgym.run import run
  from dbgym.config import get_config

  config = get_config()

  # provide path to your dataset
  config.dataset.dir = 'dataset_path'
  config.dataset.name = 'dataset_1'
  config.dataset.query = 'target.x1'
  # (optional) set additional customized configs
  config.merge_from_file('config_gcn.yaml')
  stats = run(config)

Finally, DBGym will generate experiment logs and predictions in the `output` directory by default. The predictions for column `x1` of `target.csv` will be saved as `target_pred.csv`.

Alternatively, you can customize your output directory by adding the following line

.. code-block:: bash

  config.dataset.dir = 'output_path'


3 Include customized models to DBGym
------------------------------------

You can further easily register your customized models to DBGym and run experiments with them. This is especially helpful if you want to benchmark your proposed model with DBGym. 
You may follow the instructions below to customize your own model and potentially contribute back to the DBGym repository via pull requests.

To start, instead of using the simple `pip install dbgym` method, you should:

.. code-block:: bash

  git clone https://github.com/JiaxuanYou/DBGym.git
  pip uninstall dbgym  # uninstall existing DBGym
  pip install -e .  # install with developer mode


Then, you can customize your own model in `dbgym/contribute/your_model.py`:

.. code-block:: bash

  from dbgym.register import register

  class YourModel(nn.Module):
      ...

  your_model_type = 'tabular_model' or 'graph_model'
  register(your_model_type, 'your_model_name', YourModel)


Next, you can use DBGym with your customized model

.. code-block:: bash

  from dbgym.run import run
  from dbgym.config import get_config

  config = get_config()
  config.model.name = 'your_model_name'
  stats = run(config)

