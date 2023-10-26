Example DBGym use case
======================

DBGym makes Machine Learning on Relational Databases simple.
In the current release, DBGym focuses on relational databases (databases with multiple tables), which can be as simple as a directory of CSV or Parquet files. Here, we provide a simple example of what DBGym can help you achieve.


You may have the following CSV files in your directory `database`, each representing a table:

.. code-block:: bash

   database
   ├── user.csv
   ├── item.csv
   └── trans.csv

For example, `user.csv` may look like this:

.. list-table:: User Table
   :header-rows: 1
   :widths: 10 10 10 20

   * - _user
     - x
     - y
     - time
   * - 0
     - 0.1
     - A
     - 2021/01/01
   * - 1
     - 0.8
     - A
     - 2022/02/01
   * - 2
     - 0.5
     - B
     - 2023/05/01
   * - 3
     - 0.5
     - N/A
     - 2024/08/01

Your goal is to leverage the full database (not just the `user.csv` table) to predict the unknown **N/A** values in column `y` in `user.csv`.

Let's pause here - the simple capability of predicting unknown column values is, in fact, extremely expressive. For example, it can be used for:

- Classification or regression tasks.
- Time series prediction tasks, assuming the **N/A** values are associated with future time column values.
- Recommendation tasks by predicting with respect to pairs of user and item IDs.
- Unifying node, edge, and graph-level prediction tasks on graphs by representing (heterogeneous) graphs as multiple tables.

This is a non-exhaustive list. By being creative, we can do many things with the capability of predicting unknown column values.
