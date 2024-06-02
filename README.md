# BMKINT_269 Model v1.0

The Risk Intelligence model is an ML classification model trained to identify suspicious customers and reference them for further verification, created with scikit learn's **RandomForestClassifier**.

## Usage

The model has two modules to improve running times and speed accordingly and depends on what the user wants to do, to retrain the model or to make predictions out of it.

### `RI269_training.py`

This file was used to train the ML model and create the export files which will be used into production (serialization). Should be runned when doing model retraining (previously updating the train data).

> [!NOTE]
> Note that for using this module it is recommended to create a retraining file and proceed as follows:

```python
# import the training module and its Model class 
from RI269_training import Model

# initialize a Model class instance
model = Model()

# train the model (by default, the data used should be stored in a 'data' folder 
# with the name 'train_data.xlsx' or 'train_data.csv')
model.train()

# to verify the model's cross validation score, print as follows
print(model.score)
```

### `RI269_production.py`
This file is in charge of making predictions based on the trained model and the data preprocessing stages to properly clean the dataframe.

> [!TIP]
> For easier operation, a '**deployment.py**' file will be attached to only execute this one, changing the input data.

```python
# import the production module and its Production class
from RI269_production import Production

# initialize a Production class instance
production = Production()

# run predictions
production.predict(src_path=prod_data, dataframe=prod_data)

# where prod_data can be either a source path containing the data file (csv or xlsx), or a 
# pandas dataframe. Note that only one of them should be passed as argument at a time.
```

## Predictions

In order for the predictions to run correctly, the following files must be in the same folder as the deployment file:
* `RI269_production.py`
* `trained_model.sav`

All the generated predictions will be exported to an Excel file with the format 'predict_dd-mm-yy.xlsx', including the date of program execution. In it the table will contain several columns, such as:
* customer_id
* login_name
* predicted_label
* bot_score
* wg_score
* regular_score
* manual_decision

## Requirements
A `requirements.txt` and `environment.yml` files will be loaded as well, to prevent issues when unpacking the trained model and preprocessing functions.
```
Python version used 3.11.9
```