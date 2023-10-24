[testing-image]: https://github.com/JiaxuanYou/DBGym/actions/workflows/testing.yml/badge.svg
[testing-url]: https://github.com/JiaxuanYou/DBGym/actions/workflows/testing.yml
[linting-image]: https://github.com/JiaxuanYou/DBGym/actions/workflows/linting.yml/badge.svg
[linting-url]: https://github.com/JiaxuanYou/DBGym/actions/workflows/linting.yml

<div align="center">

# DBGym
[![Testing Status][testing-image]][testing-url]
[![Linting Status][linting-image]][linting-url]

</div>

```bash
pip install dbgym	# Install DBGym
```

DBGym is a platform designed to facilitate ML research and application on databases.
With less than **5 lines of code**, you can point to your database, write the predictive query you want, and DBGym will output the predictions along with your database.

In the current release, DBGym focuses on relational databases (databases with multiple tables). In practice, it simply means that you would need to prepare your data into a directory with CSV or Parquet files, each representing a table. Then, you would need to prepare the column names in each file according to a standard naming convention: 

For example, you may have the following files in your database directory
```bash
user.csv	item.csv	trans.csv
```
Then, the column names for each file should look like
```bash
user.csv:	_user, x1, x2, ...
item.csv:	_item, x1, x2, ...
trans.csv:	_trans, _user, _item, x1, x2, ...
```
It is worth noticing that the name of files and columns should preferably without `.`, and the column name begin with `_` indicates a key value instead of feature.


# Quick Start

With less than 5 lines of code, you can set up and carryout experiments using DBGym.

```Python
from dbgym.run import run
from dbgym.config import get_config

config = get_config()
stats = run(config)
```

## Customize Configuration

For users who want to customize configuration, i.e. carry out experiment using a specific model on a specific dataset, we provide two ways to customize configuration: using code and using `yaml` file, respectively.

**Code for Configuration**

For example, we can set the target dataset using the following code:

```Python
from dbgym.run import run
from dbgym.config import get_config

config = get_config()
config.merge_from_list(['dataset.name', 'rdb1-ather'])
config.merge_from_list(['dataset.query', 'entry_examination.Cholesterol'])
stats = run(config)
```

**File for Configuration**

For example, we can set the model using the following code:

```Python
from dbgym.run import run
from dbgym.config import get_config

config = get_config()
config.merge_from_file('config_mlp.yaml')
stats = run(config)
```

And the yaml `config_mlp.yaml` is as follow:

```yaml
# Example Experiment Configuration for MLP

train:
  epoch: 200

model:
  name: 'MLP'
  hidden_dim: 128
  layer: 4

optim:
  lr: 0.001
```

## Customize Dataset

For users who want to customize dataset, i.e. carry out experiment on a given dataset, we provide the following code to customize the dataset configuration.

```Python
from dbgym.run import run
from dbgym.config import get_config

config = get_config()
config.merge_from_list(['dataset.dir', 'your_dataset_path'])
config.merge_from_list(['dataset.name', 'your_dataset_name'])
config.merge_from_list(['dataset.query', 'your_target_file.your_target_column'])
stats = run(config)
```

Where the directory of the dataset should be as follow:

```
tree your_dataset_path
your_dataset_path
├── your_dataset_name
│   ├── your_target_file
│   ├── other_file
│   └── ...
├── other_dataset
└── ...
```

## Customize Model

For users who want to customize model, i.e. carry out experiment with a novel model on our proposed dataset, we provide the following code to customize the your own model and contribute to our reposity.

Customize your own model in `dbgym/contribute/your_model.py`:

```Python
from dbgym.register import register

class YourModel(nn.Module):
    ...

your_model_type = 'tabular_model' or 'graph_model'
register(your_model_type, 'your_model_name', YourModel)
```

```Python
from dbgym.run import run
from dbgym.config import get_config

config = get_config()
config.merge_from_list(['model.name', 'your_model_name'])
stats = run(config)
```

# Highlights

DBGym has enabled RDBench, a user-friendly toolkit for benchmarking ML methods on relational databases. Please refer to our paper for more details: *[RDBench: ML Benchmark for Relational Databases](tbf)*, arXiv, Zizhao Zhang*, Yi Yang*, Lutong Zou*, He Wen*, Tao Feng, Jiaxuan You.


<div align="center">
<img align="center" src="docs/Overview.png" width="1000px" />
<b><br>Figure 1: An overview of our proposed benchmark RDBench.</b>
</div>


**1. Unified Task Definition for Various Data Formats.**
- To meet the requirements of diverse users, DBGym provides 3 kinds of data: tabular data, homogeneous graphs and heterogeneous graphs.
- For all these data formats, we propose a unified task definition, enabling results comparison between models for different formats of data.

**2. Hierarchical Datasets with Comprehensive Experiments** 
- DBGym provides 11 datasets with a range of scales, domains, and relationships. 
- These datasets are categorized into three groups based on their relationship complexity.
- Extensive experiments with 10 baselines are carried out on these datasets.

**3. Easy-to-use Interfaces with Robust Results** 
- Highly modularized pipeline.
- Reproducible experiment configuration.
- Scalable experiment management.
- Flexible user customization.
- Results reported are averaged over the same dataset and same task type (classification or regression).


# Logging
## Visualisation

To see the visualisation of training process, go into the output directory and run

```bash
tensorboard --logdir .
```

Then visit `localhost:6006`, or any other port you specified.


# Installation

DBGym is available for Python 3.10 and above.

```
pip install dbgym
```
