| Algorithm                                | Split | Train Size | Test Size | Training Time | RMSE   | R²     |
| ---------------------------------------- | ----- | ---------- | --------- | ------------- | ------ | ------ |
| LightGBM (LGBMRegressor)                 | 90/10 | 19,732     | 2,193     | 3.6s          | 2.2594 | 0.6693 |
| LightGBM (LGBMRegressor)                 | 80/20 | 17,540     | 4,385     | 2.7s          | 2.3170 | 0.6623 |
| LightGBM (LGBMRegressor)                 | 70/30 | 15,347     | 6,578     | 3.7s          | 2.3085 | 0.6436 |
| LightGBM (LGBMRegressor)                 | 50/50 | 10,962     | 10,963    | 4.1s          | 2.3197 | 0.6154 |
| Random Forest (RandomForestRegressor)    | 90/10 | 19,732     | 2,193     | 11.8s         | 2.4621 | 0.6073 |
| Random Forest (RandomForestRegressor)    | 80/20 | 17,540     | 4,385     | 4.6s          | 2.5565 | 0.5889 |
| Random Forest (RandomForestRegressor)    | 70/30 | 15,347     | 6,578     | 6.4s          | 2.5079 | 0.5794 |
| Random Forest (RandomForestRegressor)    | 50/50 | 10,962     | 10,963    | 5.6s          | 2.5063 | 0.5510 |
| Linear Regression (RidgeCV)              | 90/10 | 19,732     | 2,193     | 0.4s          | 2.6759 | 0.5362 |
| Linear Regression (RidgeCV)              | 80/20 | 17,540     | 4,385     | 0.4s          | 2.7841 | 0.5124 |
| Linear Regression (RidgeCV)              | 70/30 | 15,347     | 6,578     | 0.3s          | 2.7577 | 0.4914 |
| Linear Regression (RidgeCV)              | 50/50 | 10,962     | 10,963    | 0.2s          | 2.6865 | 0.4841 |
| Regression Trees (DecisionTreeRegressor) | 70/30 | 15,347     | 6,578     | 0.2s          | 2.8928 | 0.4404 |
| Regression Trees (DecisionTreeRegressor) | 90/10 | 19,732     | 2,193     | 0.2s          | 2.9726 | 0.4276 |
| Regression Trees (DecisionTreeRegressor) | 80/20 | 17,540     | 4,385     | 0.2s          | 3.0875 | 0.4003 |
| Regression Trees (DecisionTreeRegressor) | 50/50 | 10,962     | 10,963    | 0.1s          | 2.9811 | 0.3648 |
