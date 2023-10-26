
What DBGym Can Do for You
=========================

DBGym defines a simple declarative **predictive query** to describe the prediction task above, by specifying `table_name.column_name` in DBGym config. In the example above, the predictive query should be `user.y`, indicating that you want to set up an ML prediction task for column `y` in table `user`, based on the full database's information. DBGym will then automatically set up the ML prediction pipeline, including splitting the train/val/test sets, training an ML model, getting evaluation metrics for train/val/test sets, and returning `user_pred.csv` where any missing N/A values will be replaced with DBGym predictions.

To solve a given predictive query, DBGym provides a variety of ML models from tabular ML to graph ML communities, including: 

+-------------------+---------------------------+
|  DBGym data API  |          Model            |
+===================+===========================+
| Tabular           | XGBoost, MLP              |
+-------------------+---------------------------+
| Graph             | GCN, GraphSAGE, GIN, GAT  |
+-------------------+---------------------------+
| Heterogeneous graph | HGCN, HGT               |
+-------------------+---------------------------+


You can also easily create your own models and register them to DBGym. We hope to integrate more ML models into DBGym, and we sincerely appreciate your pull requests!

