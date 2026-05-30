| Algorithm                                | Split | Train Size | Test Size | Training Time | RMSE   | R²     |
| ---------------------------------------- | ----- | ---------- | --------- | ------------- | ------ | ------ |
| LightGBM (LGBMRegressor)                 | 80/20 | 17,529     | 4,383     | 2.4s          | 2.2097 | 0.6329 |
| LightGBM (LGBMRegressor)                 | 70/30 | 15,338     | 6,574     | 2.4s          | 2.2402 | 0.6231 |
| LightGBM (LGBMRegressor)                 | 90/10 | 19,720     | 2,192     | 2.5s          | 2.2321 | 0.6183 |
| LightGBM (LGBMRegressor)                 | 50/50 | 10,956     | 10,956    | 2.4s          | 2.2659 | 0.6147 |
| Random Forest (RandomForestRegressor)    | 80/20 | 17,529     | 4,383     | 5.3s          | 2.3756 | 0.5757 |
| Random Forest (RandomForestRegressor)    | 70/30 | 15,338     | 6,574     | 4.1s          | 2.3938 | 0.5696 |
| Random Forest (RandomForestRegressor)    | 50/50 | 10,956     | 10,956    | 2.2s          | 2.4260 | 0.5583 |
| Random Forest (RandomForestRegressor)    | 90/10 | 19,720     | 2,192     | 5.8s          | 2.4361 | 0.5453 |
| Linear Regression (RidgeCV)              | 80/20 | 17,529     | 4,383     | 0.4s          | 2.6279 | 0.4809 |
| Linear Regression (RidgeCV)              | 70/30 | 15,338     | 6,574     | 0.3s          | 2.6521 | 0.4717 |
| Linear Regression (RidgeCV)              | 50/50 | 10,956     | 10,956    | 0.2s          | 2.6538 | 0.4715 |
| Linear Regression (RidgeCV)              | 90/10 | 19,720     | 2,192     | 0.3s          | 2.6788 | 0.4502 |
| Regression Trees (DecisionTreeRegressor) | 70/30 | 15,338     | 6,574     | 0.1s          | 2.8029 | 0.4099 |
| Regression Trees (DecisionTreeRegressor) | 80/20 | 17,529     | 4,383     | 0.1s          | 2.8275 | 0.3990 |
| Regression Trees (DecisionTreeRegressor) | 50/50 | 10,956     | 10,956    | 0.1s          | 2.9033 | 0.3674 |
| Regression Trees (DecisionTreeRegressor) | 90/10 | 19,720     | 2,192     | 0.2s          | 2.8840 | 0.3628 |
