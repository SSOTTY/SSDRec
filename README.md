# SSDRec
## Input data
You can get the raw data in data folder, in which concat, dependence and interation store social relation, dependecy relation and interations between developers and software packages separately.

Besides, you can get the processed data in the PHP, ruby and JS folders, which contain following files:
- train.tsv: includes developer historical behaviors, which is organized by pandas.Dataframe in five fields (SessionId UserId ItemId Timestamps TimeId).
- valid.tsv: the same format as train.tsv, used for tuning hyperparameters.
- test.tsv: the same format as test.tsv, used for testing model.
- adj.tsv: includes links between developers, which is also organized by pandas.Dataframe in two fields (Followee, Follower, TimeId).
- dependence.tsv: includes links between packages, which is also organized by pandas.Dataframe in two fields (projectID, dependence_projectID).
- latest_session.tsv: serves as 'reference' to target developer. This file records all developers available session at each time slot. For example, at time slot t, it stores developer u's t-1 th session.
- user_id_map.tsv: maps original string developer id to int.
- item_id_map.tsv: maps original string package id to int.

## Running the code
After choosing the required input files, Run the provided script directly:

```
sh SSDRec.sh
```
