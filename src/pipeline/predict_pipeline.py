import os
import numpy as np
import pandas as pd
import tempfile
import uuid
import shutil
from src.exception import CustomException
from src.utils import load_object
import sys
sys.path.insert(0, '/src/components/data_tranformation.py')
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTrainerConfig
from typing import List
from werkzeug.datastructures import FileStorage
from sklearn.metrics import accuracy_score
from tempfile import TemporaryDirectory


# class PredictPipeline:
#     def __init__(self):
#         preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
#         model_path = os.path.join("artifacts", "model.pkl")
        
#         self.preprocessor = load_object(file_path=preprocessor_path)
#         self.model = load_object(file_path=model_path)
#         # Load the unique labels for LSTM
#         self.unique_labels_lstm = load_object('path/to/unique_labels_lstm.pkl')

        
#         self.data_transformation = DataTransformation()  # Initializing the data transformation object
#         self.transformer_methods = self.data_transformation.get_data_transformer_object()  # Getting the transformation methods
        
#         self.model_type = self.identify_model_type()  # Identify the model type (to be implemented)

#     def identify_model_type(self):
#         # Implement a mechanism to identify and return the model type
#         # This is a placeholder and you would need to implement this based on how you can identify the model type

#         # Example implementation using the class name of the model object
#         model_class_name = self.model.__class__.__name__
        
#         if 'SVM' in model_class_name:
#             return 'SVM'
#         elif 'LogisticRegression' in model_class_name:
#             return 'Logistic Regression'
#         elif 'RandomForest' in model_class_name:
#             return 'Random Forest'
#         elif 'XGB' in model_class_name:
#             return 'XGBoost'
#         elif 'Sequential' in model_class_name:  # Assuming LSTM model is a Sequential model in Keras
#             return 'LSTM'
#         else:
#             raise ValueError(f"Unknown model type: {model_class_name}")




# class CustomData:
#     def __init__(self, your_data_path=None, your_data_path_text=None):
#         self.your_data_path = your_data_path
#         self.your_data_path_text = your_data_path_text

#     def get_data_directory_path(self):
#         try:
#             if self.your_data_path:
#                 temp_dir = tempfile.mkdtemp()
#                 for file in self.your_data_path:
#                     if isinstance(file, FileStorage):
#                         filepath = os.path.join(temp_dir, file.filename)
#                         file.save(filepath)
#                 # Return the temporary directory path where files are stored
#                 return temp_dir
                       
#             elif self.your_data_path_text:
#                 # Return the directory path specified in the text input
#                 return self.your_data_path_text
                   
#         except Exception as e:
#             raise CustomException(e, sys)

# class PredictPipeline:
#     def __init__(self):
#         preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
#         self.preprocessor = load_object(file_path=preprocessor_path)
        
# class PredictPipeline:
#     def __init__(self):
#         # Initially, don't load the model. Load it after identifying its type.
#         # self.model = None

#         self.data_transformation = DataTransformation()
#         self.transformer_methods = self.data_transformation.get_data_transformer_object()

#         # This will set self.model_type attribute
#         self.model_type = self.identify_model_type()

#         # Load the model based on identified type
#         self.load_model_by_type()

        # # For LSTM labels
        # self.unique_labels_lstm = load_object('path/to/unique_labels_lstm.pkl')
class PredictPipeline:
    def __init__(self):
        self.model_types = ['Logistic Regression', 'Random Forest', 'XGBoost', 'LSTM']
        self.models = self.load_models()

        self.data_transformation = DataTransformation()
        self.transformer_methods = self.data_transformation.get_data_transformer_object()

    def load_models(self):
        config = ModelTrainerConfig()
        models = {}
        for model_type in self.model_types:
            # if model_type == 'SVM':
            #     models[model_type] = load_object(file_path=config.svm_model_file_path)
            if model_type == 'Logistic Regression':
                models[model_type] = load_object(file_path=config.logistic_model_file_path)
            if model_type == 'Random Forest':
                models[model_type] = load_object(file_path=config.rf_model_file_path)
            if model_type == 'XGBoost':
                models[model_type] = load_object(file_path=config.xgb_model_file_path)
            if model_type == 'LSTM':
                models[model_type] = load_object(file_path=config.lstm_model_file_path)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
        return models

    def predict(self, features):
        try:
            predictions = {}
            for model_type, model in self.models.items():
                # Use the appropriate data transformation method on the features before predicting
                transformer_method = self.transformer_methods['time_features']
                preprocessor_path, transformed_data = transformer_method(features)

                # Drop the 'timestamp' column
                transformed_data = transformed_data.drop(columns=['time'])

                # Reshape the data if the model is LSTM
                if model_type == 'LSTM':
                    transformed_data = transformed_data.values.reshape((transformed_data.shape[0], 1, transformed_data.shape[1]))

                # Predict using the current model after transforming the features
                preds = model.predict(transformed_data)

                # If the model is LSTM, decode the predictions from one-hot encoding
                if model_type == 'LSTM':
                    preds = np.argmax(preds, axis=1)

                predictions[model_type] = preds
            print(predictions)    
            return predictions
        except Exception as e:
            raise CustomException(e, sys)

    
class CustomData:
    def __init__(self, your_data_path=None, your_data_path_text=None):
        self.your_data_path = your_data_path
        self.your_data_path_text = your_data_path_text

    def get_data_directory_path(self):
        try:
            # If there are files in your_data_path, create a temp directory without saving files.
            if self.your_data_path:
                temp_dir = tempfile.mkdtemp()

                for file in self.your_data_path:
                    if isinstance(file, FileStorage):
                        # Do not save the file here
                        pass

                # Check if the directory was created successfully.
                if os.path.exists(temp_dir):
                    print(f"Successfully created directory: {temp_dir}")
                    # Return the temporary directory path where files can be accessed.
                    return temp_dir
                else:
                    raise ValueError("Failed to create the temporary directory.")
            
            # If no files were uploaded but a text path is provided, return that.
            if self.your_data_path_text:
                return self.your_data_path_text

            else:
                raise ValueError("No data provided for processing.")

        except Exception as e:
            raise CustomException(e, sys)


    # def get_data_directory_path(self):
    #     try:
    #         # Clean up old temporary directories
    #         for temp_directory in os.listdir(tempfile.gettempdir()):
    #             if temp_directory.startswith("tmp"):
    #                 temp_directory_path = os.path.join(tempfile.gettempdir(), temp_directory)
    #                 try:
    #                     shutil.rmtree(temp_directory_path)
    #                 except Exception as e:
    #                     print(f"Failed to delete {temp_directory_path}: {str(e)}")

    #         if self.your_data_path:
    #             # Generate a unique directory name
    #             temp_dir = os.path.join(tempfile.gettempdir(), str(uuid.uuid4()))
    #             os.makedirs(temp_dir, exist_ok=True)  # Ensures the directory exists

    #             for file in self.your_data_path:
    #                 if isinstance(file, FileStorage):
    #                     filepath = os.path.join(temp_dir, file.filename)
    #                     file.save(filepath)

    #             # Check if the directory was created successfully
    #             if os.path.exists(temp_dir):
    #                 # Return the temporary directory path where files are stored
    #                 return temp_dir
    #             else:
    #                 raise ValueError("Failed to create the temporary directory.")

    #         elif self.your_data_path_text:
    #             # Return the directory path specified in the text input
    #             return self.your_data_path_text

    #         else:
    #             raise ValueError("No data provided for processing.")

    #     except Exception as e:
    #         raise CustomException(e, sys)





   



