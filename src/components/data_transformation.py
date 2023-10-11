import sys
sys.path.append('C:\\Users\\Aeesha\\DissProject\\mlbearing')
from dataclasses import dataclass
import numpy as np 
import pandas as pd
from scipy.stats import entropy
from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object
#from src.components.model_training import ModelTrainer

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")
    csv_file_path: str = os.path.join('artifacts', "set1_timefeatures.csv")

class CustomException(Exception):
    def __init__(self, message, sys_info):
        super().__init__(message)
        self.sys_info = sys_info

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    
    def calculate_rms(self, df):
        result = []
        for col in df:
            r = np.sqrt((df[col]**2).sum() / len(df[col]))
            result.append(r)
        return result

    # extract peak-to-peak features
    def calculate_p2p(self, df):
        return np.array(df.max().abs() + df.min().abs())

    # extract shannon entropy (cut signals to 500 bins)
    def calculate_entropy(self, df):
        ent = []
        for col in df:
            ent.append(entropy(pd.cut(df[col], 500).value_counts()))
        return np.array(ent)
    # extract clearence factor
    def calculate_clearence(self, df):
        result = []
        for col in df:
            r = ((np.sqrt(df[col].abs())).sum() / len(df[col]))**2
            result.append(r)
        return result

    def time_features(self, dataset_path):
        time_features = ['mean', 'std', 'skew', 'kurtosis', 'entropy', 'rms', 'max', 'p2p', 'crest', 'clearence', 'shape', 'impulse']
        cols1 = ['B1_x', 'B1_y', 'B2_x', 'B2_y', 'B3_x', 'B3_y', 'B4_x', 'B4_y']
        cols2 = ['B1', 'B2', 'B3', 'B4']

        dataframes = []
        
        for filename in os.listdir(dataset_path):
            raw_data = pd.read_csv(os.path.join(dataset_path, filename), sep='\t')
            num_columns = len(raw_data.columns)
        
            # Calculate mean_abs, std, skew, kurtosis, etc. based on the number of columns in raw_data
            mean_abs = np.array(raw_data.abs().mean())
            std = np.array(raw_data.std())
            skew = np.array(raw_data.skew())
            kurtosis = np.array(raw_data.kurtosis())
            entropy = self.calculate_entropy(raw_data)
            rms = np.array(self.calculate_rms(raw_data))
            max_abs = np.array(raw_data.abs().max())
            p2p = self.calculate_p2p(raw_data)
            crest = max_abs / rms
            clearence = np.array(self.calculate_clearence(raw_data))
            shape = rms / mean_abs
            impulse = max_abs / mean_abs

           
            # Reshape and create DataFrames based on num_features
            if num_columns == 8:
                columns = [c+'_'+tf for c in cols1 for tf in time_features]
                data = pd.DataFrame(columns=columns)
                mean_abs = pd.DataFrame(mean_abs.reshape(1,8), columns=[c+'_mean' for c in cols1])
                std = pd.DataFrame(std.reshape(1,8), columns=[c+'_std' for c in cols1])
                skew = pd.DataFrame(skew.reshape(1,8), columns=[c+'_skew' for c in cols1])
                kurtosis = pd.DataFrame(kurtosis.reshape(1,8), columns=[c+'_kurtosis' for c in cols1])
                entropy = pd.DataFrame(entropy.reshape(1,8), columns=[c+'_entropy' for c in cols1])
                rms = pd.DataFrame(rms.reshape(1,8), columns=[c+'_rms' for c in cols1])
                max_abs = pd.DataFrame(max_abs.reshape(1,8), columns=[c+'_max' for c in cols1])
                p2p = pd.DataFrame(p2p.reshape(1,8), columns=[c+'_p2p' for c in cols1])
                crest = pd.DataFrame(crest.reshape(1,8), columns=[c+'_crest' for c in cols1])
                clearence = pd.DataFrame(clearence.reshape(1,8), columns=[c+'_clearence' for c in cols1])
                shape = pd.DataFrame(shape.reshape(1,8), columns=[c+'_shape' for c in cols1])
                impulse = pd.DataFrame(impulse.reshape(1,8), columns=[c+'_impulse' for c in cols1])

            else:
                columns = [c+'_'+tf for c in cols2 for tf in time_features]
                data = pd.DataFrame(columns=columns)
                mean_abs = pd.DataFrame(mean_abs.reshape(1,4), columns=[c+'_mean' for c in cols2])
                std = pd.DataFrame(std.reshape(1,4), columns=[c+'_std' for c in cols2])
                skew = pd.DataFrame(skew.reshape(1,4), columns=[c+'_skew' for c in cols2])
                kurtosis = pd.DataFrame(kurtosis.reshape(1,4), columns=[c+'_kurtosis' for c in cols2])
                entropy = pd.DataFrame(entropy.reshape(1,4), columns=[c+'_entropy' for c in cols2])
                rms = pd.DataFrame(rms.reshape(1,4), columns=[c+'_rms' for c in cols2])
                max_abs = pd.DataFrame(max_abs.reshape(1,4), columns=[c+'_max' for c in cols2])
                p2p = pd.DataFrame(p2p.reshape(1,4), columns=[c+'_p2p' for c in cols2])
                crest = pd.DataFrame(crest.reshape(1,4), columns=[c+'_crest' for c in cols2])
                clearence = pd.DataFrame(clearence.reshape(1,4), columns=[c+'_clearence' for c in cols2])
                shape = pd.DataFrame(shape.reshape(1,4), columns=[c+'_shape' for c in cols2])
                impulse = pd.DataFrame(impulse.reshape(1,4), columns=[c+'_impulse' for c in cols2])

        
            # Set index
            mean_abs.index = [filename]
            std.index = [filename]
            skew.index = [filename]
            kurtosis.index = [filename]
            entropy.index = [filename]
            rms.index = [filename]
            max_abs.index = [filename]
            p2p.index = [filename]
            crest.index = [filename]
            clearence.index = [filename]
            shape.index = [filename]
            impulse.index = [filename]

            # Concatenate and update data DataFrame
            merge = pd.concat([mean_abs, std, skew, kurtosis, entropy, rms, max_abs, p2p, crest, clearence, shape, impulse], axis=1)
            dataframes.append(merge)
            data = pd.concat(dataframes, ignore_index=False)
            #print('dframe', data)
        # After you've processed all files, now determine which set of columns to use based on the last raw_data
        if len(data.columns) == 96:
            cols = [c + '_' + tf for c in cols1 for tf in time_features]
            data = data[cols]
            #print('data', data)
        else:
            cols = [c + '_' + tf for c in cols2 for tf in time_features]
            data = data[cols]

        data.index = pd.to_datetime(data.index, format='%Y.%m.%d.%H.%M.%S')      
        data = data.sort_index()
        data.reset_index(inplace=True)
        data.rename(columns={'index': 'time'}, inplace=True)
        # print('sortd', data)
        data.columns = [col.replace('_x', '') for col in data.columns]

        cols_to_drop = [col for col in data.columns if '_y' in col]
        data.drop(columns=cols_to_drop, inplace=True)

        data.to_csv(self.data_transformation_config.csv_file_path, index=False, header=True)
        # print('test', data)
        return (
                self.data_transformation_config.preprocessor_obj_file_path,
                data
        )

    def get_data_transformer_object(self):
        return {
            'time_features': self.time_features
        }

    # def initiate_data_transformation(self, dataset_path):
    #     try:
            # Calculate time features using the time_features method
            # transformed_data = self.time_features(dataset_path)
            #print('tdata', transformed_data)

            # Renaming DataFrame
            # transformed_data.columns = [col.replace('_x', '') for col in transformed_data.columns]
            
            # Removing columns with '_y'
            # cols_to_drop = [col for col in transformed_data.columns if '_y' in col]
            # transformed_data.drop(columns=cols_to_drop, inplace=True)

            # Save the transformed DataFrame
            
            # transformed_data.to_csv(self.data_transformation_config.csv_file_path, index=False, header=True)
            #print('dtrans', transformed_data)

            # preprocessing_obj = self.get_data_transformer_object()
            # save_object(
            #     file_path=self.data_transformation_config.preprocessor_obj_file_path,
            #     obj=preprocessing_obj
            # )

        #     return (
        #         self.data_transformation_config.preprocessor_obj_file_path,
        #         transformed_data  # Include the transformed data in the return value
        #     )
        # except Exception as e:
        #     raise CustomException(e, sys)

if __name__ == "__main__":
    data_transformation = DataTransformation()
    directory_path = 'C:/Users/Aeesha/DissProject/mlbearing/notebook/bearing/1st_test/1st_test'
    
    ct = data_transformation.time_features(directory_path)
    # cb = ModelTrainer()
    # print(cb.initiate_model_trainer(ct))
