import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer

# load dataset
filePath = 'opa_properties_public.csv'
data = pd.read_csv(filePath, low_memory=False)

# preprocessing: handle missing values
data.dropna(subset=['sale_price', 'number_of_bathrooms', 'number_of_bedrooms',
                    'number_stories', 'interior_condition', 'exterior_condition'], inplace=True)

# select relevant columns
selected_columns = ['zip_code', 'total_area', 'number_of_bathrooms', 'number_of_bedrooms',
                     'number_stories', 'interior_condition', 'exterior_condition', 'sale_date',
                     'sale_price', 'market_value']

# create working copy with selected features and one hot encoding
selected = data[selected_columns].copy()
selected = pd.get_dummies(selected, columns=['zip_code'], drop_first=True)

# convert sale_date into datetime object, then drop it
selected['sale_date'] = pd.to_datetime(selected['sale_date'], errors='coerce')
selected.drop('sale_date', axis=1, inplace=True)

# Feature engineering
# ratio of bedrooms to bathrooms
selected['bath_bed_ratio'] = selected['number_of_bathrooms'] / selected['number_of_bedrooms']
# combining interior and exterior conditions
selected['combined_condition'] = selected['interior_condition'] + selected['exterior_condition']

# prepare x and y for model training; target is market value
x = selected.drop('market_value', axis=1)
y = selected['market_value']

# handle infinite values
x.replace([np.inf, -np.inf], np.nan, inplace=True)

# impute missing values, including infinite, using median
imputer = SimpleImputer(strategy='median')
x_imputed = imputer.fit_transform(x)
x_imputed_df = pd.DataFrame(x_imputed, columns=x.columns)

# split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_imputed_df, y, test_size=0.1, random_state=42)

# linear regression
lr_model = LinearRegression()
lr_model.fit(x_train, y_train)
lr_y_pred = lr_model.predict(x_test)
lr_mse = mean_squared_error(y_test, lr_y_pred)
lr_r2 = r2_score(y_test, lr_y_pred)
print(f"Linear Regression MSE: {lr_mse}, r^2: {lr_r2}")

# decision tree
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(x_train, y_train)
dt_y_pred = dt_model.predict(x_test)
dt_mse = mean_squared_error(y_test, dt_y_pred)
dt_r2 = r2_score(y_test, dt_y_pred)
print(f"Decision Tree MSE: {dt_mse}, r^2: {dt_r2}" )

# random forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(x_train, y_train)
rf_y_pred = rf_model.predict(x_test)
rf_mse = mean_squared_error(y_test, rf_y_pred)
rf_r2 = r2_score(y_test, rf_y_pred)
print(f"Random Forest MSE: {rf_mse}, r^2: {rf_r2}")

# gradient boosting
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb_model.fit(x_train, y_train)
gb_y_pred = gb_model.predict(x_test)
gb_mse = mean_squared_error(y_test, gb_y_pred)
gb_r2 = r2_score(y_test, gb_y_pred)
print(f"Gradient Boosting MSE: {gb_mse}, r^2: {gb_r2}")

# evaluate chosen model with mae and rmse: gradient boosting
gb_y_pred = gb_model.predict(x_test)
gb_mae = mean_absolute_error(y_test, gb_y_pred)
gb_rmse = np.sqrt(mean_squared_error(y_test, gb_y_pred))

print(f"Gradient Boosting MAE: {gb_mae}")
print(f"Gradient Boosting RMSE: {gb_rmse}")

