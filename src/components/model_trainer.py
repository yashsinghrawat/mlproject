import os
import sys

from dataclasses import dataclass
from catboost import CatBoostRegressor

from sklearn.ensemble import (

    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor

from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:

    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array,test_array):
        
        try:

            logging.info("Splitting training and test input data")
            x_train = train_array[:,:-1]
            y_train = train_array[:,-1]

            x_test = test_array[:, :-1]
            y_test = test_array[: , -1]


            models = {

                'Random Forest' : RandomForestRegressor(),
                'Decision Tree' : DecisionTreeRegressor(),
                'Gradient Boosting': GradientBoostingRegressor(),
                'Linear Regression': LinearRegression(),
                'K - Neighbors Regressor' : KNeighborsRegressor(),
                'XGB Regressor' : XGBRegressor(),
                'Cat Boosting Regressor': CatBoostRegressor(verbose = False),
                'AdaBoost Regressor':  AdaBoostRegressor(),
            }

            model_report : dict = evaluate_models(x_train , y_train,
                                                 x_test,y_test , models)

            # to get best model score from dict
            
            best_model_score = max(model_report.values())

            # to gt best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No Model can be said as 'BEST' ",sys)
            
            logging.info("Best model found on both training and testing dataset")

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj  =  best_model
            )

            predicted = best_model.predict(x_test)

            r2 = r2_score(y_test, predicted)
            return r2
            
        except Exception as e:
            raise CustomException(e,sys)