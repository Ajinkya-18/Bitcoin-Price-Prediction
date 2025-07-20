from utils import load_data, preprocess_data, train_model, save_model, test_model

df = load_data(r'data\bitcoin_2017_to_2023.csv')
print(df.shape)

df = preprocess_data(df)
