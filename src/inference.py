from utils import get_inference_data, load_model, preprocess_data


# get inference data
df = get_inference_data()
# print(df.shape)

# process the inference data
df_processed = preprocess_data(df, mode='inference')
print(df_processed)

# load a pretrained model
model = load_model('models/lr_fitted.joblib')

# make predictions on inference data
y_hat = model.predict(df_processed)

# show the predicted results
if y_hat[0] == 1.0:
    print('Prediction:\nBitcoin Price will go up.')

if y_hat[0] == 0.0:
    print('Prediction:\nBitcoin Price will not go up.')

else:
    raise ValueError('Invalid predicted value.')



