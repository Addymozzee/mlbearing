from data_transformation import DataTransformation, DataTransformationConfig, CustomException
from src.components.model_training import ModelTrainer

if __name__ == "__main__":
    data_transformation = DataTransformation()
    directory_path = 'C:/Users/Aeesha/DissProject/mlbearing/notebook/bearing/1st_test/1st_test'
    
    ct = data_transformation.initiate_data_transformation(directory_path)
    cb = ModelTrainer() # Initialize and possibly use the ModelTrainer here as necessary
