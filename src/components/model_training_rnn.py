import os
import logging
import pandas as pd
import numpy as np
import optuna
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import sys
sys.path.append('C:\\Users\\Aeesha\\DissProject\\mlbearing')
import time
import optuna
from optuna.trial import Trial
import scipy
from scipy.stats import entropy
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,classification_report
from sklearn import preprocessing
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import logging
import sys



class CustomException(Exception):
    pass

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.B1, self.B2, self.B3, self.B4 = self._initialize_dicts()

    def _initialize_dicts(self):
        B1 ={
            "early" : ["2003-10-22 12:06:24" , "2003-10-23 09:14:13"],
            "suspect" : ["2003-10-23 09:24:13" , "2003-11-08 12:11:44"],
            "normal" : ["2003-11-08 12:21:44" , "2003-11-19 21:06:07"],
            "suspect_1" : ["2003-11-19 21:16:07" , "2003-11-24 20:47:32"],
            "imminent_failure" : ["2003-11-24 20:57:32","2003-11-25 23:39:56"]
        }
        B2 = {
            "early" : ["2003-10-22 12:06:24" , "2003-11-01 21:41:44"],
            "normal" : ["2003-11-01 21:51:44" , "2003-11-24 01:01:24"],
            "suspect" : ["2003-11-24 01:11:24" , "2003-11-25 10:47:32"],
            "imminient_failure" : ["2003-11-25 10:57:32" , "2003-11-25 23:39:56"]
        }

        B3 = {
            "early" : ["2003-10-22 12:06:24" , "2003-11-01 21:41:44"],
            "normal" : ["2003-11-01 21:51:44" , "2003-11-22 09:16:56"],
            "suspect" : ["2003-11-22 09:26:56" , "2003-11-25 10:47:32"],
            "Inner_race_failure" : ["2003-11-25 10:57:32" , "2003-11-25 23:39:56"]
        }

        B4 = {
            "early" : ["2003-10-22 12:06:24" , "2003-10-29 21:39:46"],
            "normal" : ["2003-10-29 21:49:46" , "2003-11-15 05:08:46"],
            "suspect" : ["2003-11-15 05:18:46" , "2003-11-18 19:12:30"],
            "Rolling_element_failure" : ["2003-11-19 09:06:09" , "2003-11-22 17:36:56"],
            "Stage_two_failure" : ["2003-11-22 17:46:56" , "2003-11-25 23:39:56"]
        }  # the B4 dictionary
        return B1, B2, B3, B4

    def _get_state(self, time, dates_dict):
        for state, date_ranges in dates_dict.items():
            if time in date_ranges:
                return state
        return None



    def _assign_states(self, df):
        for col, dates_dict in zip(["B1_state", "B2_state", "B3_state", "B4_state"], [self.B1, self.B2, self.B3, self.B4]):
            df[col] = df.index.map(lambda x: self._get_state(x, dates_dict))
                    
        return df

    def initiate_model_trainer(self):
        try:
            logging.info("Split training and test input data")
            df = pd.read_csv('./notebook/data/set1_timefeatures.csv')
            set1 = df.rename(columns={'Unnamed: 0':'time'}).set_index('time')
            set1 = self._assign_states(set1)
            
            # Initialize lists outside the loop
            B1_state = list()
            B2_state = list()
            B3_state = list()
            B4_state = list()
            cnt = 0

            # Loop over the time index of the DataFrame
            for _ in set1.index:
                cnt += 1
                # B1
                if cnt <= 151:
                    B1_state.append("early")
                elif 151 < cnt <= 600:
                    B1_state.append("suspect")
                elif 600 < cnt <= 1499:
                    B1_state.append("normal")
                elif 1499 < cnt <= 2098:
                    B1_state.append("suspect")
                elif 2098 < cnt <= 2156:
                    B1_state.append("imminent_failure")
                else:
                    B1_state.append(None)

                # B2
                if cnt <= 500:
                    B2_state.append("early")
                elif 500 < cnt <= 2000:
                    B2_state.append("normal")
                elif 2000 < cnt <= 2120:
                    B2_state.append("suspect")
                elif 2120 < cnt <= 2156:
                    B2_state.append("imminet_failure")
                else:
                    B2_state.append(None)

                # B3
                if cnt <= 500:
                    B3_state.append("early")
                elif 500 < cnt <= 1790:
                    B3_state.append("normal")
                elif 1790 < cnt <= 2120:
                    B3_state.append("suspect")
                elif 2120 < cnt <= 2156:
                    B3_state.append("Inner_race_failure")
                else:
                    B3_state.append(None)

                # B4
                if cnt <= 200:
                    B4_state.append("early")
                elif 200 < cnt <= 1000:
                    B4_state.append("normal")
                elif 1000 < cnt <= 1435:
                    B4_state.append("suspect")
                elif 1435 < cnt <= 1840:
                    B4_state.append("Inner_race_failure")
                elif 1840 < cnt <= 2156:
                    B4_state.append("Stage_two_failure")
                else:
                    B4_state.append(None)

            # Assign lists to DataFrame columns after the loop completes
            set1["B1_state"] = B1_state
            set1["B2_state"] = B2_state
            set1["B3_state"] = B3_state
            set1["B4_state"] = B4_state


            # for col, dates_dict in zip(["B1_state", "B2_state", "B3_state", "B4_state"], [self.B1, self.B2, self.B3, self.B4]):
            #     set1[col] = set1.index.map(lambda x: self._get_state(x, dates_dict))
                
            B1_cols = [col for col in set1.columns if "B1" in col]
            B2_cols = [col for col in set1.columns if "B2" in col]
            B3_cols = [col for col in set1.columns if "B3" in col]
            B4_cols = [col for col in set1.columns if "B4" in col]

            B1 = set1[B1_cols]
            B2 = set1[B2_cols]
            B3 = set1[B3_cols]
            B4 = set1[B4_cols]
            cols = ['Bx_mean','Bx_std','Bx_skew','Bx_kurtosis','Bx_entropy','Bx_rms','Bx_max','Bx_p2p','Bx_crest', 'Bx_clearence', 'Bx_shape', 'Bx_impulse',
                        'By_mean','By_std','By_skew','By_kurtosis','By_entropy','By_rms','By_max','By_p2p','By_crest', 'By_clearence', 'By_shape', 'By_impulse',
                        'class']
            # cols = [... ]  # The same list of column names as before
            B1.columns = cols
            B2.columns = cols
            B3.columns = cols
            B4.columns = cols
            final_data = pd.concat([B1, B2, B3, B4], axis=0, ignore_index=True)

            X = final_data.copy()
            y = X.pop("class")
            le = preprocessing.LabelEncoder()
            y = le.fit_transform(y)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
            
            # Reshape the data for LSTM
            X_train = np.reshape(X_train.values, (X_train.shape[0], 1, X_train.shape[1]))
            X_test = np.reshape(X_test.values, (X_test.shape[0], 1, X_test.shape[1]))

            return(X_train, X_test, y_train, y_test)

        except Exception as e:
            raise CustomException(str(e))

        # For RNN 
        # Modifying the objective_lstm function

    def objective_lstm(self, trial):
        start_time = time.time()
        
        X_train, X_test, y_train, y_test = self.initiate_model_trainer()

        n_units = trial.suggest_int("n_units", 16, 128, log=True)
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5, step=0.1)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)

        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(units=n_units, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
            tf.keras.layers.Dropout(rate=dropout_rate),
            tf.keras.layers.LSTM(units=n_units),
            tf.keras.layers.Dropout(rate=dropout_rate),
            tf.keras.layers.Dense(units=len(np.unique(y_train)), activation='softmax')
        ])

        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            metrics=["accuracy"]
        )

        model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=0)

        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)

        accuracy = accuracy_score(y_test, y_pred_classes)

        end_time = time.time()
        self.log_and_print("LSTM", accuracy, end_time - start_time)

        return accuracy

    def log_and_print(self, model_name, score, duration):
        logging.info(f"Model {model_name} found with score: {score}")
        logging.info(f"Model {model_name} took {duration:.2f} seconds to train.")
        print(f"Model {model_name} found with score: {score:.4f}")
        print(f"Model {model_name} took {duration:.2f} seconds to train.")

if __name__ == "__main__":
    vibd = ModelTrainer()

    # Optimize for LSTM
    study_lstm = optuna.create_study(direction='maximize')
    study_lstm.optimize(vibd.objective_lstm, n_trials=20)  # 20 trials, adjust this as needed

    print("\n--- LSTM Results ---")
    print("Best Parameters: ", study_lstm.best_params)
    print("Best Score: ", study_lstm.best_value)
    logging.info("Best Parameters: %s", study_lstm.best_params)
    logging.info("Best Score: %s", study_lstm.best_value)







   





