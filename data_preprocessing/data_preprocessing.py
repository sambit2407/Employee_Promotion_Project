from application_logging import App_Logger
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from data_loader import data_getter_training


class Preprocessing:

    def __init__(self,file_object,logger_object):
        self.logger_object = logger_object
        self.file_object = file_object

    def droping_unnecessary_cols(self,data):
        try:

            self.data = data.drop(columns='employee_id', axis=1)


            self.logger_object.log(self.file_object,
                               'column has been removed')
            return self.data
        except Exception as e:


            self.logger_object.log(self.file_object,
                               'Error in droping_unnecessary_cols :  ' + str(
                                   e))

    def checking_treating_missing_values(self,data):
        self.logger_object.log(self.file_object, 'Entered the is_null_present method of the Preprocessor class')
        self.cols = data.columns
        self.null_counts = data.isna().sum()
        self.null_cols = []
        self.con_cols = ['age','avg_training_score']
        self.cat_cols = ['department', 'region', 'education', 'gender', 'recruitment_channel',
       'no_of_trainings', 'previous_year_rating', 'length_of_service',
       'KPIs_met', 'awards_won',
       'previous_year_rating_Var']
        self.data=data
        try:
            if data.isnull().values.any():
                for i in range(len(self.null_counts)):
                    if self.null_counts[i] > 0:
                        self.null_cols.append(self.cols[i])

                for i in self.null_cols:
                    if i in self.con_cols:

                        data[i] = data[i].replace(np.NAN, data[i].mean())


                    else:
                        self.data['previous_year_rating_Var'] = np.where(self.data['previous_year_rating'].isnull(), 1, 0)

                        data[i] = data[i].fillna(data[i].value_counts().index[0])

            self.logger_object.log(self.file_object,
                                   'Data has been treated')


        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in is_null_present method of the Preprocessor class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Finding missing values failed. Exited the is_null_present method of the Preprocessor class')
            raise Exception()

    def outier_treatment(self, data, columns=None):

        self.logger_object.log(self.file_object, 'Entered the outlier treatment method of the Preprocessor class')

        self.data = data
        self.col = columns

        try:
            for col in self.col:

                self.uppper_boundary = data[col].mean() + 3 * data[col].std()
                self.lower_boundary = data[col].mean() - 3 * data[col].std()
                # print(lower_boundary), print(uppper_boundary), print(data['age'].mean())
                self.logger_object.log(self.file_object,
                                       f'Upper limit :{self.uppper_boundary}, lower limit:{self.lower_boundary} set for column {col}')
                try:
                    data.loc[data[col] > self.uppper_boundary, col] = self.uppper_boundary
                except Exception as e:
                    self.logger_object.log(self.file_object, 'getting error %s' % e)

                self.logger_object.log(self.file_object, f'Column {col} has been treated.')



        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in Outlier TReatment Method of the Preprocessor class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Outlier treatment failed. Exited the is_null_present method of the Preprocessor class')
            return 'error: %s' % e

    def scaling_of_numcol(self,data):
        """
                                                                Method Name: scaling_of_Numcol
                                                                Description: This method scales the numerical values using the Standard scaler.
                                                                Output: A dataframe with scaled values
                                                                On Failure: Raise Exception

                                                                Written By: sambit kumar behera
                                                                Version: 1.0
                                                                Revisions: None
                                             """
        self.logger_object.log(self.file_object,
                               'Entered the scaling_of_Numcol method of the Preprocessor class')

        self.data = data
        self.num_df = self.data[['age','avg_training_score']]

        try:

            self.scaler = StandardScaler()
            self.scaled_data = self.scaler.fit_transform(self.num_df)
            self.scaled_num_df = pd.DataFrame(data=self.scaled_data, columns=self.num_df.columns, index=self.data.index)
            self.data.drop(columns=self.scaled_num_df.columns, inplace=True)
            self.data = pd.concat([self.scaled_num_df, self.data], axis=1)

            self.logger_object.log(self.file_object,
                                   'scaling for numerical values successful. Exited the scale_numerical_columns method of the Preprocessor class')
            return self.data

        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in scale_numerical_columns method of the Preprocessor class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'scaling for numerical columns Failed. Exited the scale_numerical_columns method of the Preprocessor class')
            raise Exception()
    def target_dependable_encode_fordeparmrnt(self):
        data_getter = data_getter_training.Data_Getter_Training(self.file_object, self.logger_object)
        data = data_getter.get_data()


        self.ordinal_labels = data.groupby(['department'])['is_promoted'].mean().sort_values().index



    def encode_categorical_col(self,data):
        """
                                                Method Name: encode_categorical_col
                                                Description: This method encodes the categorical values to numeric values.
                                                Output: dataframe with categorical values converted to numerical values
                                                On Failure: Raise Exception

                                                Written By:Sambit umar behera Intelligence
                                                Version: 1.0
                                                Revisions: None
                             """
        self.logger_object.log(self.file_object, 'Entered the encode_categorical_columns method of the Preprocessor class')

        self.data=data
        try:



            ordinal_labels2 = {k: i for i, k in enumerate(self.ordinal_labels, 0)}
            self.data['department'] = data['department'].map(ordinal_labels2)
            self.logger_object.log(self.file_object,
                                   'treating department column successfull')

            region_map = self.data['region'].value_counts().to_dict()
            self.data['region'] = self.data['region'].map(region_map)
            self.logger_object.log(self.file_object,
                                   'treating region column successfull')

            self.data['education'] = data['education'].replace(
                {"Master's & above": 3, "Bachelor's": 2, 'Below Secondary': 1})
            self.logger_object.log(self.file_object,
                                   'treating education column successfull')

            self.data = pd.get_dummies(data, drop_first=True)
            self.logger_object.log(self.file_object,
                                   'treating getting dummies for gender column successfull')


            return self.data

        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in encode_categorical_columns method of the Preprocessor class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object, 'encoding for categorical columns Failed. Exited the encode_categorical_columns method of the Preprocessor class')
            raise Exception()

    def separate_label_feature(self, data, label_column_name):
        self.logger_object.log(self.file_object, 'Entered the separate_label_feature method of the Preprocessor class')
        try:
            self.X = data.drop(labels=label_column_name,
                               axis=1)  # drop the columns specified and separate the feature columns
            self.Y = data[label_column_name]  # Filter the Label columns
            self.logger_object.log(self.file_object,
                                   'Label Separation Successful. Exited the separate_label_feature method of the Preprocessor class')
            return self.X, self.Y
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in separate_label_feature method of the Preprocessor class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Label Separation Unsuccessful. Exited the separate_label_feature method of the Preprocessor class')
            raise Exception()


