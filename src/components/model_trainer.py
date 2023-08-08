import os
import sys
sys.path.append('C:\\Users\\Aeesha\\DissProject\\mlbearing')
import pandas as pd
# import scipy
# from scipy.stats import entropy
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score,f1_score,classification_report
from sklearn import preprocessing
import xgboost as xgb
import catboost as cb
import lightgbm as lgbm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from sklearn.svm import SVC
# import plotly.express as px
# import plotly.graph_objects as go
# from src.utils import save_object,evaluate_models
# import optuna
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import ConfusionMatrixDisplay
# from collections import Counter
# from sklearn.preprocessing import StandardScaler
# from sklearn.neural_network import MLPClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.gaussian_process import GaussianProcessClassifier
# from sklearn.gaussian_process.kernels import RBF
# from sklearn.metrics import accuracy_score
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier



@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,X,y):
        try:
            logging.info("Split training and test input data")
            
            set1 = self.rename(columns={'Unnamed: 0':'time'})
            set1.set_index('time')

            B1 ={
            "no_failure" : ["2003-10-22 12:06:24" , "2003-10-23 09:14:13",
                            "2003-10-23 09:24:13" , "2003-11-08 12:11:44",
                            "2003-11-08 12:21:44" , "2003-11-19 21:06:07",
                            "2003-11-19 21:16:07" , "2003-11-24 20:47:32"],
            "failure" : ["2003-11-24 20:57:32" , "2003-11-25 23:39:56"]
                    }
            B2 = {
            "no_failure" : ["2003-10-22 12:06:24" , "2003-11-01 21:41:44",
                            "2003-11-01 21:51:44" , "2003-11-24 01:01:24",
                            "2003-11-24 01:11:24" , "2003-11-25 10:47:32"],
            "failure" : ["2003-11-25 10:57:32" , "2003-11-25 23:39:56"]
                    }

            B3 = {
            "no_failure" : ["2003-10-22 12:06:24" , "2003-11-01 21:41:44",
                            "2003-11-01 21:51:44" , "2003-11-22 09:16:56",
                            "2003-11-22 09:26:56" , "2003-11-25 10:47:32"],
            "failure" : ["2003-11-25 10:57:32" , "2003-11-25 23:39:56"]
                    }

            B4 = {
            "no_failure" : ["2003-10-22 12:06:24" , "2003-10-29 21:39:46",
                            "2003-10-29 21:49:46" , "2003-11-15 05:08:46",
                            "2003-11-15 05:18:46" , "2003-11-18 19:12:30"],
            "failure" : ["2003-11-19 09:06:09" , "2003-11-22 17:36:56",
                        "2003-11-22 17:46:56" , "2003-11-25 23:39:56"]
                    }
            
            B1_state = list()
            B2_state = list()
            B3_state = list()
            B4_state = list()
            cnt = 0

            for row in set1["time"]:
                cnt += 1
                # B1
                if cnt<=2098:
                    B1_state.append("no_failure")
                if 2098 < cnt <= 2156:
                    B1_state.append("failure")
                #B2
                if cnt<=2120:
                    B2_state.append("no_failure")
                if 2120< cnt <=2156:
                    B2_state.append("failure")

                #B3
                if cnt<=2120:
                    B3_state.append("no_failure")
                if 2120 < cnt <=2156:
                    B3_state.append("failure")
                #B4
                if cnt<=1435:
                    B4_state.append("no_failure")
                if 1435 < cnt <=2156:
                    B4_state.append("failure")
                #controlling the counts
                set1["B1_state"] = B1_state
                set1["B2_state"] = B2_state
                set1["B3_state"] = B3_state
                set1["B4_state"] = B4_state


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
            B1.columns = cols
            B2.columns = cols
            B3.columns = cols
            B4.columns = cols
            final_data = pd.concat([B1,B2,B3,B4], axis=0, ignore_index=True)

            X = final_data.copy()
            y = X.pop("class")
            le = preprocessing.LabelEncoder() #le is a variable name that you can change and change the 'class' to numerical value
            le.fit(y)
            y = le.transform(y)
            X_x_train, X_x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state =1)
            
            # time_features_list = ["mean","std","skew","kurtosis","entropy","rms","max","p2p", "crest", "clearence", "shape", "impulse"]
            # x_axis_cols = ["Bx_"+tf for tf in time_features_list]
            # X_x = X.copy()
            # X_x = X[x_axis_cols]
            # cols = ['B_mean','B_std','B_skew','B_kurtosis','B_entropy',
            #         'B_rms','B_max','B_p2p','B_crest', 'B_clearence', 'B_shape', 'B_impulse']
            # X_x.columns = cols
            # X_x_train, X_x_test, y_train, y_test = train_test_split(X_x, y, test_size = 0.3, random_state =1)

            names = ["Nearest Neighbors", "Linear SVM", "RBF SVM",
                    #"Gaussian Process",
                    "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
                    "Naive Bayes", "QDA","XGBoost","CatGBoost","LightGBoost"]

            classifiers = [
                # KNeighborsClassifier(3),
                SVC(kernel="linear", C=0.025),
                SVC(gamma=2, C=1),
                # GaussianProcessClassifier(1.0 * RBF(1.0)),
                # DecisionTreeClassifier(max_depth=5),
                RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
                # MLPClassifier(alpha=1, max_iter=1000),
                AdaBoostClassifier(),
                GaussianNB(),
                QuadraticDiscriminantAnalysis(),
                xgb.XGBClassifier(),
                cb.CatBoostClassifier(verbose = False),
                lgbm.LGBMClassifier()
                ]

            for name, clf in zip(names, classifiers):
                    print("Training " + name + "...")
                    clf.fit(X_x_train, y_train)
                    predicted_labels = clf.predict(X_x_test)
                    score = clf.score(X_x_test, y_test)
                    labels = ['N', 'F']
            if not (score and labels):
                raise CustomException("No best model found")
            logging.info("Best model found for both training and testing datasets", score)


            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=score
            )

            return predicted_labels


        except Exception as e:
            raise CustomException(e,sys)


    # def initiate_model_trainer(self,train_array,test_array):
    #     try:
    #         logging.info("Split training and test input data")
    #         X_train,y_train,X_test,y_test=(
    #             train_array[:,:-1],
    #             train_array[:,-1],
    #             test_array[:,:-1],
    #             test_array[:,-1]
    #         )
#             models = {
#                 "Random Forest": RandomForestRegressor(),
#                 "Support Vector Machine": SVR(),
#                 "Gradient Boosting": GradientBoostingRegressor(),
#                 "Logistic Regression": LogisticRegression(),
#                 "XGBRegressor": XGBRegressor(),
#                 "CatBoosting Regressor": CatBoostRegressor(verbose=False),
#                 "AdaBoost Regressor": AdaBoostRegressor(),
#             }
#             params={
#                 "Decision Tree": {
#                     'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
#                     # 'splitter':['best','random'],
#                     # 'max_features':['sqrt','log2'],
#                 },
#                 "Random Forest":{
#                     # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
#                     # 'max_features':['sqrt','log2',None],
#                     'n_estimators': [8,16,32,64,128,256]
#                 },
#                 "Gradient Boosting":{
#                     # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
#                     'learning_rate':[.1,.01,.05,.001],
#                     'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
#                     # 'criterion':['squared_error', 'friedman_mse'],
#                     # 'max_features':['auto','sqrt','log2'],
#                     'n_estimators': [8,16,32,64,128,256]
#                 },
#                 "Linear Regression":{},
#                 "XGBRegressor":{
#                     'learning_rate':[.1,.01,.05,.001],
#                     'n_estimators': [8,16,32,64,128,256]
#                 },
#                 "CatBoosting Regressor":{
#                     'depth': [6,8,10],
#                     'learning_rate': [0.01, 0.05, 0.1],
#                     'iterations': [30, 50, 100]
#                 },
#                 "AdaBoost Regressor":{
#                     'learning_rate':[.1,.01,0.5,.001],
#                     # 'loss':['linear','square','exponential'],
#                     'n_estimators': [8,16,32,64,128,256]
#                 }
                
#             }

#             model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
#                                              models=models,param=params)
            
#             ## To get best model score from dict
#             best_model_score = max(sorted(model_report.values()))

#             ## To get best model name from dict

#             best_model_name = list(model_report.keys())[
#                 list(model_report.values()).index(best_model_score)
#             ]
#             best_model = models[best_model_name]

#             if best_model_score<0.6:
#                 raise CustomException("No best model found")
#             logging.info(f"Best found model on both training and testing dataset")

#             save_object(
#                 file_path=self.model_trainer_config.trained_model_file_path,
#                 obj=best_model
#             )

#             predicted=best_model.predict(X_test)

#             r2_square = r2_score(y_test, predicted)
#             return r2_square
            



            
#         except Exception as e:
#             raise CustomException(e,sys)