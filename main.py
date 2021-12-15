import pickle
import pandas as pd
from Modules.data_processors import DataPreprocessor, FeatureGenerator

# loading dataframes
data_test_df = pd.read_csv("data_test.csv")
features_df = pd.DataFrame(pd.read_csv('./Features/features.csv', sep="\t"))

# loading model and feature generator attributes
with open('Modules/model.pkl', 'rb') as pkl: model = pickle.load(pkl) 
with open('Modules/feature_gen.pkl', 'rb') as pikl: feature_generator = pickle.load(pikl) 

# initialize Data Preprocessor and Feature Generator, set Feature Generator arrtibutes from learned instance
data_preprocessor = DataPreprocessor(data_test_df, features_df)
feature_gen = FeatureGenerator()
feature_gen.__dict__ = feature_generator

# transform data
merged_test = data_preprocessor.transform()
merged_test = feature_gen.transform(merged_test)

# predict with model
predictions = model.predict(merged_test)

# merge predictions with dataframe, dump dataframe to csv
data_test_df['target'] = predictions
data_test_df.to_csv('answers_test.csv', sep='\t', columns=['buy_time', 'id', 'vas_id', 'target'], index=False)
