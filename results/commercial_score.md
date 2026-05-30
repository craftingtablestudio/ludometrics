| Algorithm                                | Split | Train Size | Test Size | Training Time | RMSE    | R²     |
| ---------------------------------------- | ----- | ---------- | --------- | ------------- | ------- | ------ |
| LightGBM (LGBMRegressor)                 | 90/10 | 19,720     | 2,192     | 2.5s          | 8.4613  | 0.6331 |
| LightGBM (LGBMRegressor)                 | 80/20 | 17,529     | 4,383     | 2.5s          | 8.6812  | 0.6184 |
| LightGBM (LGBMRegressor)                 | 70/30 | 15,338     | 6,574     | 2.4s          | 8.7202  | 0.6126 |
| LightGBM (LGBMRegressor)                 | 50/50 | 10,956     | 10,956    | 2.3s          | 8.8282  | 0.5997 |
| Random Forest (RandomForestRegressor)    | 90/10 | 19,720     | 2,192     | 5.6s          | 9.1926  | 0.5670 |
| Random Forest (RandomForestRegressor)    | 80/20 | 17,529     | 4,383     | 4.8s          | 9.3897  | 0.5536 |
| Random Forest (RandomForestRegressor)    | 70/30 | 15,338     | 6,574     | 3.6s          | 9.4122  | 0.5486 |
| Random Forest (RandomForestRegressor)    | 50/50 | 10,956     | 10,956    | 2.5s          | 9.5290  | 0.5336 |
| Linear Regression (RidgeCV)              | 90/10 | 19,720     | 2,192     | 0.3s          | 10.1199 | 0.4752 |
| Linear Regression (RidgeCV)              | 80/20 | 17,529     | 4,383     | 0.3s          | 10.2422 | 0.4689 |
| Linear Regression (RidgeCV)              | 70/30 | 15,338     | 6,574     | 0.2s          | 10.2410 | 0.4656 |
| Linear Regression (RidgeCV)              | 50/50 | 10,956     | 10,956    | 0.2s          | 10.2135 | 0.4642 |
| Regression Trees (DecisionTreeRegressor) | 90/10 | 19,720     | 2,192     | 0.2s          | 10.4652 | 0.4388 |
| Regression Trees (DecisionTreeRegressor) | 80/20 | 17,529     | 4,383     | 0.1s          | 10.5383 | 0.4377 |
| Regression Trees (DecisionTreeRegressor) | 70/30 | 15,338     | 6,574     | 0.1s          | 10.5313 | 0.4349 |
| Regression Trees (DecisionTreeRegressor) | 50/50 | 10,956     | 10,956    | 0.1s          | 10.7013 | 0.4118 |
