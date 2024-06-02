## IMPORTS ##


import datetime
import joblib
import os
import numpy as np
import pandas as pd



## MODEL ##


class Production():
    def __init__(self) -> None:
        """
        Model definition and initialization
        """        
        (self.model, 
         self.encoder, 
         self.scaler, 
         self.columns, 
         self.ct) = joblib.load('trained_model.sav')
        

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Raw data pulling and preprocessing from source path or dataframe

        Args:
            df (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Clean dataframe for predictions
        """        
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
        
        df[cities] = df[cities].apply(lambda x: x.str.title())
        df[cities] = df[cities].replace(['-', '.'], np.nan)
        df[cities] = df[cities].replace(regex=r'\d+', value=np.nan)
        
        df['gender'] = df['gender'].fillna(value='MALE')
        
        df['morning_ratio'] = df['morning_tickets'] / df['daytime_tickets']
        df['night_ratio'] = df['night_tickets'] / df['daytime_tickets']
        
        df = df.drop(drop_cols, axis=1).drop_duplicates(
            subset='customer_id', 
            ignore_index=True
        ).reset_index(drop=True)

        df = df.set_index(['customer_id', 'login_name'])
        df = df[self.columns]
        
        categorical = df.select_dtypes(exclude='number')
        numerical = df.select_dtypes(include='number')
        
        X = self.ct.transform(df)
        X.columns = self.columns
        
        return X


    def predict(self, src_path: str='', dataframe: pd.DataFrame=pd.DataFrame()) -> None:
        """
        Given a source path or dataframe, predicts labels 
        and export an excel file with the suspicious ones

        Args:
            src_path (str, optional): Input source path containing csv or excel file. Defaults to ''.
            dataframe (pd.DataFrame, optional): Pandas dataframe. Defaults to pd.DataFrame().
        """        
        if src_path:
            if src_path[-4:] == 'xlsx':
                df = pd.read_excel(src_path)
            else:
                df = pd.read_csv(src_path)
        elif not dataframe.empty:
            df = dataframe
        else:
            raise ValueError('Expected at least one argument, DataFrame or Source Path')
            
        columns = [
            'customer_id',
            'login_name',
            'predicted_label',
            'bot_score',
            'wg_score',
            'regular_score',
            'manual_decision'
        ]
        rows = {
            0: 'Bot',
            1: 'WG',
            2: 'Regular'
        }
        
        X = self.preprocess(df)
        indexes = X.reset_index()[['customer_id', 'login_name']]
        
        today = datetime.date.today().strftime("%d-%m-%y")

        rfc_pred = pd.DataFrame(self.model.predict(X))
        rfc_pred_prob = pd.DataFrame(self.model.predict_proba(X))

        comparison = pd.concat(
            [
                indexes, 
                rfc_pred, 
                rfc_pred_prob
            ], 
            axis=1
        )
        comparison['manual_decision'] = [None] * comparison.shape[0]

        comparison.columns = columns
        comparison['predicted_label'] = comparison['predicted_label'].map(rows)

        suspicious_cst = comparison[
            (comparison['predicted_label'] != 'Regular') & 
            (comparison['regular_score'] <= 0.2)
        ].sort_values(by='regular_score')
        
        if not os.path.exists(os.path.join(os.getcwd(), 'model_predictions')):
            os.mkdir('model_predictions')
        
        suspicious_cst.to_excel(
            f"model_predictions/predict_{today}.xlsx",
            index=False
        )


if __name__ == '__main__':
    production = Production()
    
    production.predict(src_path='data/omitted_data.xlsx')
