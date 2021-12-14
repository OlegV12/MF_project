import pickle
import pandas as pd
from Model.data_classes import DataPreprocessor, FeatureGenerator

data_test_df = pd.read_csv("data_test.csv")
model = pickle.load(open('./Model/model.pkl', 'rb'))
features_df = pd.DataFrame(pd.read_csv('./Features/features.csv', sep="\t"))
feature_generator = pickle.load(open('./Model/feature_gen.pkl', 'rb'))

data_preprocessor = DataPreprocessor(data_test_df, features_df)

feature_gen = FeatureGenerator()
feature_gen.__dict__ = feature_generator.__dict__.copy()


merged_test = data_preprocessor.transform()
merged_test = feature_gen.transform(merged_test)


predictions = model.predict(merged_test)

data_test_df['target'] = predictions

data_test_df.to_csv('answers_test.csv', sep='\t', columns=['buy_time', 'id', 'vas_id', 'target'], index=False)

