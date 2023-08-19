import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    pred_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, pred_val)
    return mae


melbourne_file_path = 'melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)

# dropna drops missing values (think of na as "not available", reads "drop not available (data)")
melbourne_data = melbourne_data.dropna(axis=0)

# Pull out prices with dot-notation, single column stored in Series
# This is the column we want to predict, which is called the prediction target (called y by convention)
y = melbourne_data.Price

# Features are data columns inputted into an ML model that are used to make predictions (called X by convention)
# Select multiple features by storing them in a list
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_features]

# Visually checking your data with these commands is an important part of a data scientist's job.
# You'll frequently find surprises in the dataset that deserve further inspection.
print(X.describe())
print(X.head())

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

# Define model. Specify a number for random_state to ensure same results each run
melbourne_model = DecisionTreeRegressor(random_state=0)

# Fit model
melbourne_model.fit(train_X, train_y)

# Make predictions
print("The predictions are")
predictions = melbourne_model.predict(val_X)
print(predictions)
print(f'The mean absolute error is ${mean_absolute_error(val_y, predictions):.2f}')

# compare MAE with differing values of max_leaf_nodes
candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
min_mae = 10**100
best_tree_size = -1
# Write loop to find the ideal tree size from candidate_max_leaf_nodes
for max_leaf in candidate_max_leaf_nodes:
    mae = get_mae(max_leaf, train_X, val_X, train_y, val_y)
    print(f'{max_leaf} -> ${mae}')
    if min_mae >= mae:
        min_mae = mae
        best_tree_size = max_leaf

print(best_tree_size)
final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size)
final_model.fit(X, y)

# RandomForestRegressor makes more accurate predictions than DecisionTreeRegressor
forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, preds))
