| Algorithm                                | Split | Train Size | Test Size | Training Time | RMSE    | R²     |
| ---------------------------------------- | ----- | ---------- | --------- | ------------- | ------- | ------ |
| LightGBM (LGBMRegressor)                 | 90/10 | 19,732     | 2,193     | 5.3s          | 8.7447  | 0.6380 |
| LightGBM (LGBMRegressor)                 | 80/20 | 17,540     | 4,385     | 4.0s          | 8.7772  | 0.6352 |
| LightGBM (LGBMRegressor)                 | 80/20 | 17,540     | 4,385     | 2.8s          | 8.7772  | 0.6352 |
| LightGBM (LGBMRegressor)                 | 70/30 | 15,347     | 6,578     | 3.7s          | 8.8475  | 0.6204 |
| LightGBM (LGBMRegressor)                 | 50/50 | 10,962     | 10,963    | 3.6s          | 8.8286  | 0.6092 |
| Random Forest (RandomForestRegressor)    | 90/10 | 19,732     | 2,193     | 9.7s          | 9.5172  | 0.5713 |
| Random Forest (RandomForestRegressor)    | 80/20 | 17,540     | 4,385     | 7.1s          | 9.5508  | 0.5681 |
| Random Forest (RandomForestRegressor)    | 80/20 | 17,540     | 4,385     | 4.4s          | 9.5508  | 0.5681 |
| Random Forest (RandomForestRegressor)    | 70/30 | 15,347     | 6,578     | 9.1s          | 9.6000  | 0.5531 |
| Random Forest (RandomForestRegressor)    | 50/50 | 10,962     | 10,963    | 6.9s          | 9.5918  | 0.5387 |
| Linear Regression (RidgeCV)              | 90/10 | 19,732     | 2,193     | 0.3s          | 10.2609 | 0.5016 |
| Linear Regression (RidgeCV)              | 80/20 | 17,540     | 4,385     | 0.3s          | 10.3130 | 0.4964 |
| Linear Regression (RidgeCV)              | 80/20 | 17,540     | 4,385     | 0.3s          | 10.3130 | 0.4964 |
| Linear Regression (RidgeCV)              | 70/30 | 15,347     | 6,578     | 0.3s          | 10.3414 | 0.4814 |
| Linear Regression (RidgeCV)              | 50/50 | 10,962     | 10,963    | 0.2s          | 10.2321 | 0.4751 |
| Regression Trees (DecisionTreeRegressor) | 90/10 | 19,732     | 2,193     | 0.2s          | 10.6837 | 0.4597 |
| Regression Trees (DecisionTreeRegressor) | 80/20 | 17,540     | 4,385     | 0.2s          | 10.7135 | 0.4565 |
| Regression Trees (DecisionTreeRegressor) | 70/30 | 15,347     | 6,578     | 0.1s          | 10.8196 | 0.4323 |
| Regression Trees (DecisionTreeRegressor) | 50/50 | 10,962     | 10,963    | 0.1s          | 10.8029 | 0.4149 |
