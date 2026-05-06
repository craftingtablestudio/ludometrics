| Algorithm                                | Split | Train Size | Test Size | Training Time | RMSE   | R²     |
| ---------------------------------------- | ----- | ---------- | --------- | ------------- | ------ | ------ |
| LightGBM (LGBMRegressor)                 | 80/20 | 17,540     | 4,385     | 2.6s          | 2.3170 | 0.6623 |
| Random Forest (RandomForestRegressor)    | 80/20 | 17,540     | 4,385     | 8.0s          | 2.5565 | 0.5889 |
| Linear Regression (RidgeCV)              | 80/20 | 17,540     | 4,385     | 0.3s          | 2.7841 | 0.5124 |
| Regression Trees (DecisionTreeRegressor) | 80/20 | 17,540     | 4,385     | 0.2s          | 3.0875 | 0.4003 |
