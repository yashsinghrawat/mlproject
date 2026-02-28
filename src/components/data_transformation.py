import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path= os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:

    def __init__(self):
        self.data_transformation_config= DataTransformationConfig()

    def get_data_transformer_object(self):

        '''
        This function is responsible for data transformation
        
        '''
        try:
            numerical_features = ['writing score','reading score']
            categorical_features = [
                'gender',
                'race/ethnicity',
                'parental level of education',
                'lunch',
                'test preparation course',
                ]
            
            numerical_pipeline = Pipeline(
                steps = [

                ('imputer',SimpleImputer(strategy = 'median')),
                ('scaler',StandardScaler())

                ]
            )
            
            categorical_pipeline = Pipeline(

                steps = [

                    ('imputer',SimpleImputer(strategy = 'most_frequent')),
                    ('OneHotEncoder',OneHotEncoder(handle_unknown= 'ignore',sparse_output= False)),
                    ('scaler',StandardScaler())
                         
                ]

            )

            logging.info(f'Numerical Columns : {numerical_features}')

            logging.info(f'Categorical Columns : {categorical_features}')

            preprocessor= ColumnTransformer(
                [

                    ('numerical_pipeline',numerical_pipeline,numerical_features),
                    ( 'categorical_pipeline',categorical_pipeline,categorical_features  )

                ]
            )

            return preprocessor
        
        except Exception as e :
            
            raise CustomException(e,sys)

    
    def initiate_data_transformation(self,train_path,test_path):
        
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info('Obtaining preprocessing object')

            preprocessor_obj = self.get_data_transformer_object()
            target_column = 'math score'

            numerical_features = ['writing score','reading score']
            

            input_feature_train = train_df.drop(columns = [target_column])
            target_feature_train = train_df[target_column]

            input_feature_test = test_df.drop(columns = [target_column])
            target_feature_test = test_df[target_column]


            logging.info(

                f"Applying preprocessing object on training dataframe and testing dataframe"
            
            )

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test)

            train_arr = np.c_[

                input_feature_train_arr ,np.array(target_feature_train)

            ]

            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test)
            ]

            logging.info(f"Saved Preprocessing Object")



            save_object( # used for saving the pkl file

                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj
            )

            return (

                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )            
        except Exception as e:

            raise CustomException(e,sys)
