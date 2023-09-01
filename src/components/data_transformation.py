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
# from sklearn.compose import ColumnTransformer
# from sklearn.impute import SimpleImputer
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import OneHotEncoder, StandardScaler


# @dataclass
# class DataTransformationConfig:
#     preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")
#     csv_file_path: str=os.path.join('artifacts',"set1_timefeatures.csv")

# class DataTransformation:
#     def __init__(self):
#         self.data_transformation_config = DataTransformationConfig()

#     def get_data_transformer_object(self):

#         # Root Mean Squared Sum
#         def calculate_rms(self, df):
#             result = []
#             for col in df:
#                 r = np.sqrt((df[col]**2).sum() / len(df[col]))
#                 result.append(r)
#             return result

#         # extract peak-to-peak features
#         def calculate_p2p(self, df):
#             return np.array(df.max().abs() + df.min().abs())

#         # extract shannon entropy (cut signals to 500 bins)
#         def calculate_entropy(self, df):
#             ent = []
#             for col in df:
#                 ent.append(entropy(pd.cut(df[col], 500).value_counts()))
#             return np.array(ent)
#         # extract clearence factor
#         def calculate_clearence(self, df):
#             result = []
#             for col in df:
#                 r = ((np.sqrt(df[col].abs())).sum() / len(df[col]))**2
#                 result.append(r)
#             return result
        
        # def time_features(self, dataset_path, id_set=None):
        #     time_features = ['mean','std','skew','kurtosis','entropy','rms','max','p2p', 'crest', 'clearence', 'shape', 'impulse']
        #     cols1 = ['B1_x','B1_y','B2_x','B2_y','B3_x','B3_y','B4_x','B4_y']
        #     cols2 = ['B1','B2','B3','B4']

        #     # initialize
        #     # Adding columns name for time features
        #     if id_set == 1:
        #         columns = [c+'_'+tf for c in cols1 for tf in time_features]
        #         data = pd.DataFrame(columns=columns)
        #     else:
        #         columns = [c+'_'+tf for c in cols2 for tf in time_features]
        #         data = pd.DataFrame(columns=columns)

        #     for raw_data in os.listdir(dataset_path):
        #         # read dataset
        #         raw_data = pd.read_csv(os.path.join(dataset_path, raw_data), sep='\t')
                
        #         # time features
        #         mean_abs = np.array(raw_data.abs().mean())
        #         std = np.array(raw_data.std())
        #         skew = np.array(raw_data.skew())
        #         kurtosis = np.array(raw_data.kurtosis())
        #         entropy = calculate_entropy(raw_data)
        #         rms = np.array(calculate_rms(raw_data))
        #         max_abs = np.array(raw_data.abs().max())
        #         p2p = calculate_p2p(raw_data)
        #         crest = max_abs/rms
        #         clearence = np.array(calculate_clearence(raw_data))
        #         shape = rms / mean_abs
        #         impulse = max_abs / mean_abs

        #         if id_set == 1:
        #             mean_abs = pd.DataFrame(mean_abs.reshape(1,8), columns=[c+'_mean' for c in cols1])
        #             std = pd.DataFrame(std.reshape(1,8), columns=[c+'_std' for c in cols1])
        #             skew = pd.DataFrame(skew.reshape(1,8), columns=[c+'_skew' for c in cols1])
        #             kurtosis = pd.DataFrame(kurtosis.reshape(1,8), columns=[c+'_kurtosis' for c in cols1])
        #             entropy = pd.DataFrame(entropy.reshape(1,8), columns=[c+'_entropy' for c in cols1])
        #             rms = pd.DataFrame(rms.reshape(1,8), columns=[c+'_rms' for c in cols1])
        #             max_abs = pd.DataFrame(max_abs.reshape(1,8), columns=[c+'_max' for c in cols1])
        #             p2p = pd.DataFrame(p2p.reshape(1,8), columns=[c+'_p2p' for c in cols1])
        #             crest = pd.DataFrame(crest.reshape(1,8), columns=[c+'_crest' for c in cols1])
        #             clearence = pd.DataFrame(clearence.reshape(1,8), columns=[c+'_clearence' for c in cols1])
        #             shape = pd.DataFrame(shape.reshape(1,8), columns=[c+'_shape' for c in cols1])
        #             impulse = pd.DataFrame(impulse.reshape(1,8), columns=[c+'_impulse' for c in cols1])

        #         else:
        #             mean_abs = pd.DataFrame(mean_abs.reshape(1,4), columns=[c+'_mean' for c in cols2])
        #             std = pd.DataFrame(std.reshape(1,4), columns=[c+'_std' for c in cols2])
        #             skew = pd.DataFrame(skew.reshape(1,4), columns=[c+'_skew' for c in cols2])
        #             kurtosis = pd.DataFrame(kurtosis.reshape(1,4), columns=[c+'_kurtosis' for c in cols2])
        #             entropy = pd.DataFrame(entropy.reshape(1,4), columns=[c+'_entropy' for c in cols2])
        #             rms = pd.DataFrame(rms.reshape(1,4), columns=[c+'_rms' for c in cols2])
        #             max_abs = pd.DataFrame(max_abs.reshape(1,4), columns=[c+'_max' for c in cols2])
        #             p2p = pd.DataFrame(p2p.reshape(1,4), columns=[c+'_p2p' for c in cols2])
        #             crest = pd.DataFrame(crest.reshape(1,4), columns=[c+'_crest' for c in cols2])
        #             clearence = pd.DataFrame(clearence.reshape(1,4), columns=[c+'_clearence' for c in cols2])
        #             shape = pd.DataFrame(shape.reshape(1,4), columns=[c+'_shape' for c in cols2])
        #             impulse = pd.DataFrame(impulse.reshape(1,4), columns=[c+'_impulse' for c in cols2])

        #         mean_abs.index = [raw_data]
        #         std.index = [raw_data]
        #         skew.index = [raw_data]
        #         kurtosis.index = [raw_data]
        #         entropy.index = [raw_data]
        #         rms.index = [raw_data]
        #         max_abs.index = [raw_data]
        #         p2p.index = [raw_data]
        #         crest.index = [raw_data]
        #         clearence.index = [raw_data]
        #         shape.index = [raw_data]
        #         impulse.index = [raw_data]

        #         # concat
        #         merge = pd.concat([mean_abs, std, skew, kurtosis, entropy, rms, max_abs, p2p,crest,clearence, shape, impulse], axis=1)
        #         data = pd.concat([data, merge])

        #     if id_set == 1:
        #         cols = [c+'_'+tf for c in cols1 for tf in time_features]
        #         data = data[cols]
        #     else:
        #         cols = [c+'_'+tf for c in cols2 for tf in time_features]
        #         data = data[cols]

        #     data.index = pd.to_datetime(data.index, format='%Y.%m.%d.%H.%M.%S')
        #     data = data.sort_index()
        #     return data

#     def initiate_data_transformation(self, df=None):
#         try:
#             df = pd.read_csv('./notebook/bearing/1st_test/1st_test/2003.10.22.12.06.24', sep='\t')
            
#             # Remove columns containing '_y'
#             cols_to_drop = [col for col in df.columns if '_y' in col]
#             df.drop(columns=cols_to_drop, inplace=True)
            
#             # Renaming DataFrame
#             df.rename(columns=lambda x: x.replace('_x', ''), inplace=True)
#             df.rename(columns={df.columns[0]: 'time'}, inplace=True)
            
#             # Saving modified DataFrame
#             #csv_file_path = 'set1_timefeatures.csv'
#             #df.to_csv(csv_file_path, index=False)  # Save the DataFrame
#             df.to_csv(self.data_transformation_config.csv_file_path,index=False,header=True)
            
#             #logging.info(f"Modified DataFrame saved as '{csv_file_path}'.")

#             # Save preprocessing object
#             preprocessing_obj = self.get_data_transformer_object()  # Replace this with your actual preprocessing object
#             save_object(
#                 file_path=self.data_transformation_config.preprocessor_obj_file_path,
#                 obj=preprocessing_obj
#             )
            
#             return (
#                 self.data_transformation_config.preprocessor_obj_file_path,
#             )
#         except Exception as e:
#             raise CustomException(e, sys)

# if __name__=="__main__":
#     # obj=DataIngestion()
#     # train_data,test_data=obj.initiate_data_ingestion()

#     data_transformation = DataTransformation()
#     preprocessor_obj_file_path = data_transformation.initiate_data_transformation()


#     # modeltrainer=ModelTrainer()
#     # print(modeltrainer.initiate_model_trainer(train_data,test_data))

# def calculate_rms(self, series):
    #     return np.sqrt((series**2).sum() / len(series))

    # def calculate_p2p(self, series):
    #     return series.max() - series.min()

    # def calculate_entropy(self, series):
    #     entropy_values = {}
    #     for col in series.columns:
    #         entropy_values[col] = entropy(pd.cut(series[col], 500).value_counts())
    #     return pd.Series(entropy_values)

    # def calculate_clearence(self, series):
    #     return ((np.sqrt(series.abs())).sum() / len(series))**2

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

        data = pd.DataFrame()  # Create an empty DataFrame to store the results for each file

        for filename in os.listdir(dataset_path):
            file_path = os.path.join(dataset_path, filename)
            raw_data = pd.read_csv(file_path, sep='\t')
        # # Read raw data
        # dataset_path = 'C:/Users/Aeesha/DissProject/mlbearing/notebook/bearing/1st_test/1st_test/2003.10.22.12.06.24'
        # raw_data = pd.read_csv(dataset_path, sep='\t')  # You should read the raw data here

            # Determine which columns to use based on this raw_data
            num_columns = len(raw_data.columns)
            if num_columns == 8:
                columns_to_use = cols1
            elif num_columns == 4:
                columns_to_use = cols2
            else:
                raise ValueError("Unsupported number of columns in raw data")
        
        # Initialize DataFrame with column names for time features
        columns = [c + '_' + tf for c in columns_to_use for tf in time_features]
        data = pd.DataFrame(columns=columns)
        
        # Calculate mean_abs, std, skew, kurtosis, etc. based on the number of columns in raw_data
        num_columns = len(raw_data.columns)
        if num_columns == len(cols1):
            num_features = 8
        elif num_columns == len(cols2):
            num_features = 4
        else:
            raise ValueError("Unsupported number of columns in raw data")

        # for filename in os.listdir(dataset_path):
        #     # read dataset
        #     raw_data = pd.read_csv(os.path.join(dataset_path, filename), sep='\t')

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
            if num_features == 8:
                mean_abs = pd.DataFrame(mean_abs.reshape(1, 8), columns=[c + '_mean' for c in columns_to_use], index=[columns_to_use[0]])
                std = pd.DataFrame(std.reshape(1, 8), columns=[c + '_std' for c in columns_to_use], index=[columns_to_use[0]])
                skew = pd.DataFrame(skew.reshape(1, 8), columns=[c + '_skew' for c in columns_to_use], index=[columns_to_use[0]])
                kurtosis = pd.DataFrame(kurtosis.reshape(1, 8), columns=[c + '_kurtosis' for c in columns_to_use], index=[columns_to_use[0]])
                entropy = pd.DataFrame(entropy.reshape(1, 8), columns=[c + '_entropy' for c in columns_to_use], index=[columns_to_use[0]])
                rms = pd.DataFrame(rms.reshape(1, 8), columns=[c + '_rms' for c in columns_to_use], index=[columns_to_use[0]])
                max_abs = pd.DataFrame(max_abs.reshape(1, 8), columns=[c + '_max' for c in columns_to_use], index=[columns_to_use[0]])
                p2p = pd.DataFrame(p2p.reshape(1, 8), columns=[c + '_p2p' for c in columns_to_use], index=[columns_to_use[0]])
                crest = pd.DataFrame(crest.reshape(1, 8), columns=[c + '_crest' for c in columns_to_use], index=[columns_to_use[0]])
                clearence = pd.DataFrame(clearence.reshape(1, 8), columns=[c + '_clearence' for c in columns_to_use], index=[columns_to_use[0]])
                shape = pd.DataFrame(shape.reshape(1, 8), columns=[c + '_shape' for c in columns_to_use], index=[columns_to_use[0]])
                impulse = pd.DataFrame(impulse.reshape(1, 8), columns=[c + '_impulse' for c in columns_to_use], index=[columns_to_use[0]])
                # mean_abs = pd.DataFrame(mean_abs.reshape(1, 8), columns=[c + '_mean' for c in columns_to_use], index=columns_to_use)
                # std = pd.DataFrame(std.reshape(1, 8), columns=[c + '_std' for c in columns_to_use], index=columns_to_use)
            else:
                mean_abs = pd.DataFrame(mean_abs.reshape(1, 4), columns=[c + '_mean' for c in columns_to_use], index=[columns_to_use[0]])
                std = pd.DataFrame(std.reshape(1, 4), columns=[c + '_std' for c in columns_to_use], index=[columns_to_use[0]])
                skew = pd.DataFrame(skew.reshape(1, 4), columns=[c + '_skew' for c in columns_to_use], index=[columns_to_use[0]])
                kurtosis = pd.DataFrame(kurtosis.reshape(1, 4), columns=[c + '_kurtosis' for c in columns_to_use], index=[columns_to_use[0]])
                entropy = pd.DataFrame(entropy.reshape(1, 4), columns=[c + '_entropy' for c in columns_to_use], index=[columns_to_use[0]])
                rms = pd.DataFrame(rms.reshape(1, 4), columns=[c + '_rms' for c in columns_to_use], index=[columns_to_use[0]])
                max_abs = pd.DataFrame(max_abs.reshape(1, 4), columns=[c + '_max' for c in columns_to_use], index=[columns_to_use[0]])
                p2p = pd.DataFrame(p2p.reshape(1, 4), columns=[c + '_p2p' for c in columns_to_use], index=[columns_to_use[0]])
                crest = pd.DataFrame(crest.reshape(1, 4), columns=[c + '_crest' for c in columns_to_use], index=[columns_to_use[0]])
                clearence = pd.DataFrame(clearence.reshape(1, 4), columns=[c + '_clearence' for c in columns_to_use], index=[columns_to_use[0]])
                shape = pd.DataFrame(shape.reshape(1, 4), columns=[c + '_shape' for c in columns_to_use], index=[columns_to_use[0]])
                impulse = pd.DataFrame(impulse.reshape(1, 4), columns=[c + '_impulse' for c in columns_to_use], index=[columns_to_use[0]])

            
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
            merge = pd.concat([...], axis=1)
            data = pd.concat([data, merge])

        # After you've processed all files, now determine which set of columns to use based on the last raw_data
        if len(raw_data.columns) == 8:
            cols = [c + '_' + tf for c in cols1 for tf in time_features]
            data = data[cols]
        else:
            cols = [c + '_' + tf for c in cols2 for tf in time_features]
            data = data[cols]

        data.index = pd.to_datetime(data.index, format='%Y.%m.%d.%H.%M.%S')
        data = data.sort_index()

        return data

    def get_data_transformer_object(self):
        return {
            'time_features': self.time_features
        }

    def initiate_data_transformation(self, dataset_path):
        try:
            # Calculate time features using the time_features method
            transformed_data = self.time_features(dataset_path)

            # Renaming DataFrame
            transformed_data.columns = [col.replace('_x', '') for col in transformed_data.columns]
            transformed_data.rename(columns={transformed_data.columns[0]: 'time'}, inplace=True)

            # Removing columns with '_y'
            cols_to_drop = [col for col in transformed_data.columns if '_y' in col]
            transformed_data.drop(columns=cols_to_drop, inplace=True)

            # Save the transformed DataFrame
            csv_file_path = 'set1_timefeatures.csv'
            transformed_data.to_csv(self.data_transformation_config.csv_file_path, index=False, header=True)

            preprocessing_obj = self.get_data_transformer_object()
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                self.data_transformation_config.preprocessor_obj_file_path,
                transformed_data  # Include the transformed data in the return value
            )
        except Exception as e:
            raise CustomException(e, sys)
# if __name__ == "__main__":
#     data_transformation = DataTransformation()
#     ct = data_transformation.initiate_data_transformation()   
    # Call the time_features method with the dataset path
    #transformed_data = data_transformation.time_features(dataset_path)
if __name__ == "__main__":
    data_transformation = DataTransformation()
    directory_path = 'C:/Users/Aeesha/DissProject/mlbearing/notebook/bearing/1st_test/1st_test'
    ct = data_transformation.initiate_data_transformation(directory_path)
    
    # Now you have the transformed data in the 'transformed_data' DataFrame
    #print(transformed_data)