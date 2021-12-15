import pandas as pd

class DataPreprocessor():
    def __init__(self, data, features):
        self.data = data
        self.features = features

    def fit(self):
        return self


    def transform(self):
        merged = pd.merge(self.data, self.features, how='left', on=['id',])
        merged['time_delta'] = abs(merged['buy_time_x'] - merged['buy_time_y'])
        merged.sort_values(['Unnamed: 0_x', 'time_delta'], ascending=True, inplace=True)
        merged.drop_duplicates(subset=['Unnamed: 0_x'], inplace=True)

        return merged


class FeatureGenerator():
    def __init__(self) -> None:
        pass

    
    def fit(self, df):
        self.median_time = df.loc[df['target']==1].groupby(['vas_id'])['buy_time_x'].agg('median').to_dict()
        self.vas_month = df.loc[df['target']==1].groupby(['vas_id'])['buy_time_x'].agg(lambda x:x.value_counts().index[0])
        self.vas_month = pd.to_datetime(self.vas_month, unit='s').dt.month.to_dict()
        return self
    
    def transform(self, df):
        df['vas_time'] = df['vas_id'].map(self.median_time)
        df.drop(['Unnamed: 0_x', 'Unnamed: 0_y', 'id',], axis=1, inplace=True)
        df['buy_month'] = pd.to_datetime(df['buy_time_x'], unit='s').dt.month
        df = pd.get_dummies(df, columns=['vas_id'])
        df['buy_month_delta'] = df['buy_month'] - pd.to_datetime(df['vas_time']).dt.month
        df.drop(['buy_month',], axis=1, inplace=True)

        return df
