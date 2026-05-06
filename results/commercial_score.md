| Algorithm                                | Split | Train Size | Test Size | Training Time | RMSE    | R²     |
| ---------------------------------------- | ----- | ---------- | --------- | ------------- | ------- | ------ |
| LightGBM (LGBMRegressor)                 | 80/20 | 17,540     | 4,385     | 2.6s          | 8.7772  | 0.6352 |
| Random Forest (RandomForestRegressor)    | 80/20 | 17,540     | 4,385     | 6.8s          | 9.5508  | 0.5681 |
| Linear Regression (RidgeCV)              | 80/20 | 17,540     | 4,385     | 0.4s          | 10.3130 | 0.4964 |
| Regression Trees (DecisionTreeRegressor) | 80/20 | 17,540     | 4,385     | 0.2s          | 10.7135 | 0.4565 |
