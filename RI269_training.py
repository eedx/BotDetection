## IMPORTS ##

import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, OrdinalEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score

## MODEL ##

class Model():
    def __init__(self) -> None:
        """
        Model definition and initialization
        """        
        self.columns = None
        self.categorical = None
        self.numerical = None
        self.score = None
        self.ct = None
        
        self.model = RandomForestClassifier(
            max_depth=22,
            random_state=42,
            criterion='entropy',
            verbose=0
        )
        self.encoder = OrdinalEncoder(
            handle_unknown='use_encoded_value',
            unknown_value=-1,
            encoded_missing_value=-1
        )
        self.scaler = MinMaxScaler()
        
        self.train_df = self.preprocess()


    def preprocess(self, data_src: str='data/train_data.xlsx') -> pd.DataFrame:
        """
        Raw data pulling from source path and preprocessing

        Args:
            data_src (str, optional): Source path. Defaults to 'data/train_data.xlsx'.
            
        Returns:
            pd.DataFrame: Clean dataframe
        """       
        if data_src[-4:] == 'xlsx':
            df = pd.read_excel(data_src)
        else:
            df = pd.read_csv(data_src)
            
        columns = []
        for column in df.columns:
            columns.append(column.lower())
        df.columns = columns
        
        df['customer_id'] = df['customer_id'].apply(str)
        
        fill_cols = df.select_dtypes(include='number').columns.values
        drop_cols = [
            'morning_tickets',
            'daytime_tickets',
            'night_tickets'
        ]
        cities = [
            'city',
            'birthcity'
        ]
        
        for col in fill_cols:
            if col == 'daytime_tickets':
                df[col] = df[col].fillna(value=0.1)
            elif col == 'avg_quote':
                df[col] = df[col].fillna(value=1)
            else:
                df[col] = df[col].fillna(value=0)
        
        df[cities] = df[cities].replace(['-', '.'], np.nan)
        df[cities] = df[cities].apply(lambda x: x.str.title())
        df[cities] = df[cities].replace(regex=r'\d+', value=np.nan)
        
        
        df['gender'] = df['gender'].fillna(value='MALE')
        
        df['morning_ratio'] = df['morning_tickets'] / df['daytime_tickets']
        df['night_ratio'] = df['night_tickets'] / df['daytime_tickets']
        
        
        df = df.drop(drop_cols, axis=1).drop_duplicates(
            subset='customer_id', 
            ignore_index=True
        ).reset_index(drop=True)
        
        df = df.sample(frac=1, random_state=42)
        
        df['label'] = df['label'].str.lower()
        
        target = df[['customer_id', 'login_name', 'label']]
        df = df.drop('label', axis=1)
        
        y = self.target_process(target)
        
        for var in [df, y]:
            var.set_index(['customer_id', 'login_name'], inplace=True)
        
        self.columns = df.columns.values
        self.categorical = df.select_dtypes(exclude='number')
        self.numerical = df.select_dtypes(include='number')
        
        self.ct = ColumnTransformer(
            [
                ('cat_encoding', self.encoder, self.categorical.columns),
                ('num_scaling', self.scaler, self.numerical.columns)
            ],
            verbose_feature_names_out=True
        )
        self.ct.set_output(transform='pandas')
        self.ct.fit(df)
        
        X = self.ct.transform(df)
        X.columns = self.columns
        
        return X, y


    def target_process(self, targets: pd.DataFrame) -> pd.DataFrame:
        """
        Targets variable mapping into integers

        Args:
            targets (pd.DataFrame): String values from targets

        Returns:
            pd.DataFrame: Parametrized labels
        """      
        cst_dict = {
            'bot': 0,
            'wg': 1,
            'tax': 1,
            'regular': 2
        }
        
        targets['label'] = targets['label'].map(cst_dict)

        return targets
    

    def train(self) -> None:
        """
        Model training function.
        Gets all the input data preprocessed and fits the model

        Args:
            input_data (pd.DataFrame): Preprocessed input data
            targets (pd.DataFrame): Parametrized labels
        """        
        X, y = self.train_df
        
        cv_score = cross_val_score(
            self.model, 
            X, 
            y.values.ravel(), 
            cv=10
        )
        self.score = (
            f'Cross validation mean score: {cv_score.mean() * 100:0.2f}%'
        )
        
        self.model.fit(X, y.values.ravel())
        
        self.save_model(X.columns.values)
            

    def save_model(self, columns) -> None:
        """
        Model serialization function
        Creates a file containing the trained model, categorical data encoder
        and numerical data scaler in an array
        """        
        self.encoder.fit(self.categorical)
        self.scaler.fit(self.numerical)
        
        trained_model = [
            self.model, 
            self.encoder, 
            self.scaler, 
            self.columns,
            self.ct
        ]
        
        model_save = joblib.dump(trained_model, 'trained_model.sav')


if __name__ == '__main__':
    model = Model()
    model.train()

    print(model.score)
