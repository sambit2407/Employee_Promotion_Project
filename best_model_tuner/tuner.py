from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from application_logging import App_Logger
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
from data_preprocessing.data_preprocessing import Preprocessing
class Model_Finder:
    """
                This class shall  be used to find the model with best accuracy and AUC score.
                Written By: Sambit Kumar BEhera
                Version: 1.0
                Revisions: None

                """

    def __init__(self):
        self.file_object = open("../ModelFinder_Logs/model_Log.txt", 'a+')
        self.logger_object = App_Logger()
        self.rf_classifier=RandomForestClassifier()
        self.xgb_classifier = XGBClassifier()



    def get_best_params_for_randomforest(self,train_x,train_y):




        self.logger_object.log(self.file_object, 'Entered the get_best_params_for_randomforest method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            self.random_grid = {'n_estimators' : [100, 300, 500, 800, 1200],
                                'max_depth' : [5, 8, 15, 25, 30],
                                'min_samples_split' : [2, 5, 10, 15, 100],
                                'min_samples_leaf' :[1, 2, 5, 10]}

            #Creating an object of the Grid Search class
            self.grid = GridSearchCV(estimator =self.rf_classifier, param_grid = self.random_grid, cv = 3, verbose=2, n_jobs = -1)
            #finding the best parameters
            self.grid.fit(train_x, train_y)

            #extracting the best parameters
            self.n_estimators = self.grid.best_params_['n_estimators']
            self.max_depth = self.grid.best_params_['max_depth']

            self.min_samples_leaf = self.grid.best_params_['min_samples_leaf']
            self.min_samples_split = self.grid.best_params_['min_samples_split']




            #creating a new model with the best parameters
            self.rf_classifier = RandomForestClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth,min_samples_split=self.min_samples_split,min_samples_leaf=self.min_samples_leaf)


            # training the mew model
            self.rf_classifier.fit(train_x, train_y)
            self.logger_object.log(self.file_object,
                                   'Random Forest best params: '+str(self.grid.best_params_)+'. Exited the get_best_params_for_RandomForest method of the Model_Finder class')

            return self.rf_classifier
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_params_for_RandomForest method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Ramdom Forest training  failed. Exited the get_best_params_for_RandomForest method of the Model_Finder class')
            raise Exception()

    def get_best_params_for_xgboost(self,train_x,train_y):

        """
                                        Method Name: get_best_params_for_xgboost
                                        Description: get the parameters for XGBoost Algorithm which give the best accuracy.
                                                     Use Hyper Parameter Tuning.
                                        Output: The model with the best parameters
                                        On Failure: Raise Exception

                                        Written By: Sambit kumarBEhera
                                        Version: 1.0
                                        Revisions: None

                                """
        self.logger_object.log(self.file_object,
                               'Entered the get_best_params_for_xgboost method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            self.param_grid_xgboost = {

             "learning_rate": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
             "max_depth" : [ 3, 4, 5, 6, 8, 10, 12, 15],
             "min_child_weight" : [ 1, 3, 5, 7 ],
             "gamma" : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
             "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]

       }
            # Creating an object of the Grid Search class
            self.grid= GridSearchCV(estimator =self.xgb_classifier,param_grid=self.param_grid_xgboost, verbose=3,cv=5)
            # finding the best parameters
            self.grid.fit(train_x, train_y)

            # extracting the best parameters
            self.max_depth = self.grid.best_params_['max_depth']

            self.gamma = self.grid.best_params_['gamma']
            self.learning_rate = self.grid.best_params_['learning_rate']
            self.min_child_weight = self.grid.best_params_['min_child_weight']
            self.colsample_bytree = self.grid.best_params_['colsample_bytree']

            # creating a new model with the best parameters
            self.xgb = self.xgb_classifier( max_depth=self.max_depth,learning_rate=self.learning_rate ,min_child_weight=self.min_child_weight,colsample_bytree=self.colsample_bytree,n_jobs=-1)
            # training the mew model
            self.xgb.fit(train_x, train_y)
            self.logger_object.log(self.file_object,
                                   'XGBoost best params: ' + str(
                                       self.grid.best_params_) + '. Exited the get_best_params_for_xgboost method of the Model_Finder class')
            return self.xgb
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_params_for_xgboost method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'XGBoost Parameter tuning  failed. Exited the get_best_params_for_xgboost method of the Model_Finder class')
            raise Exception()

    def get_best_model(self,train_x,train_y,test_x,test_y):
        """
                                                Method Name: get_best_model
                                                Description: Find out the Model which has the best AUC score.
                                                Output: The best model name and the model object
                                                On Failure: Raise Exception

                                                Written By: Sambit kumar BEhera
                                                Version: 1.0
                                                Revisions: None

                                        """
        self.logger_object.log(self.file_object,
                               'Entered the get_best_model method of the Model_Finder class')
        # create best model for XGBoost
        try:
            self.xgboost= self.get_best_params_for_xgboost(train_x,train_y)
            prediction_xgboost = self.xgboost.predict(test_x) # Predictions using the XGBoost Model
            F1_Score_xg = metrics.f1_score(test_y, prediction_xgboost, average='weighted')

            self.rf_cls = self.get_best_params_for_randomforest(train_x, train_y)
            prediction_randomcls = self.xgboost.predict(test_x)  # Predictions using the XGBoost Model
            F1_Score_rf = metrics.f1_score(test_y, prediction_randomcls, average='weighted')



            self.logger_object.log(self.file_object,f'Score for XGBoost is: {self.F1_Score_xg}')

            self.logger_object.log(self.file_object, f'Score for RandomForest is: {self.F1_Score_rf}')




            #comparing the two models
            if(self.F1_Score_xg <  self.F1_Score_rf):
                return 'RandomForest',self.rf_cls
            else:
                return 'XGBoost',self.self.xgboost

        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_model method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Model Selection Failed. Exited the get_best_model method of the Model_Finder class')
            raise Exception()
