from application_logging import App_Logger
from data_preprocessing.data_preprocessing import Preprocessing
from file_operation import file_op
import numpy as np
import pandas as pd
import data



class Prediction_from_api:
    def __init__(self, department, region, education,gender,recruitment_channel, no_of_trainings, age, previous_year_rating,length_of_service,KPIs_met,awards_won,avg_training_score):
        self.file_object = open("prediction_logs/prediction_logs.txt", 'a+')
        self.log_writer = App_Logger()
        self.department = department
        self.region = region
        self.education = education
        self.gender = gender
        self.recruitment_channel = recruitment_channel
        self.age = age
        self.no_of_trainings = no_of_trainings
        self.previous_year_rating = previous_year_rating
        self.length_of_service = length_of_service
        self.KPIs_met = KPIs_met
        self.awards_won = awards_won
        self.avg_training_score = avg_training_score




    def prediction_api(self):
        self.log_writer.log(self.file_object, 'Start of Prediction of api....')
        InputData = pd.DataFrame(
            data=[[self.department, self.region, self.education, self.gender, self.recruitment_channel, self.age, self.no_of_trainings, self.previous_year_rating, self.length_of_service, self.KPIs_met, self.awards_won, self.avg_training_score]],
            columns=['department', 'region', 'education','gender','recruitment_channel', 'no_of_trainings', 'age', 'previous_year_rating','length_of_service','KPIs_met >80%','awards_won?','avg_training_score'])
        Num_Inputs = InputData.shape[0]
        preprocessor = Preprocessing(self.file_object, self.log_writer)
        DataForMl = pd.read_csv('data/train.csv')
        DataForMl=preprocessor.droping_unnecessary_cols(DataForMl)

        DataForMl = DataForMl.drop(columns='is_promoted', axis=1)


        InputData = InputData.append(DataForMl)

        self.log_writer.log(self.file_object, f'datafor ML: {InputData.head()}')

        predictors = ['department', 'region', 'education', 'no_of_trainings', 'age', 'previous_year_rating', 'length_of_service', 'KPIs_met >80%', 'awards_won?', 'avg_training_score', 'previous_year_rating_Var', 'gender_m', 'recruitment_channel_referred', 'recruitment_channel_sourcing']
        # Generating the input values to the model


        # print(InputData.isna().sum())
        preprocessor.checking_treating_missing_values(InputData)


        preprocessor.outier_treatment(data,columns='age')

        preprocessor.target_dependable_encode_fordeparmrnt()

        InputData = preprocessor.encode_categorical_col(InputData)

        InputData = preprocessor.scaling_of_numcol(InputData)

        X = InputData[predictors].values[0:Num_Inputs]
        self.log_writer.log(self.file_object, f'Input data for model: {X}')

        file_loader = file_op.File_Operation(self.file_object)
        xgboost = file_loader.load_model('FinalXGBModel_new')

        prediction = xgboost.predict(X)
        if prediction==1:
            return 'congratulations !! You are Promoted .'
        else:
            return 'Sorry !! You are Not Promoted .'





