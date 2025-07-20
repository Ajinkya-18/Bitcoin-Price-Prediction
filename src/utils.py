def assert_path(file_path:str):
    import os
    from pathlib import Path

    cwd = os.getcwd()
    parsed_file_path = file_path.replace('\\', '/')

    full_path = os.path.join(cwd, Path(parsed_file_path))

    if os.path.exists(full_path):
        return full_path
    

    raise ValueError('Invalid file path specified.')
    
#---------------------------------------------------------------------------------------------------------

def load_data(data_path:str):
    try:
        parsed_data_path = assert_path(data_path)

        if parsed_data_path.endswith('.csv'):
            import pandas as pd
        
            df = pd.read_csv(parsed_data_path)
            
            return df


    except Exception as e:
        raise e
    
#---------------------------------------------------------------------------------------------------------

def preprocess_data(df):
    import numpy as np

    splitted = df['timestamp'].str.split(' ', expand=True)
    df['date'] = splitted[0].astype('str')
    df['time'] = splitted[1].astype('str')

    splitted1 = df['date'].str.split('-', expand=True)
    df['year'] = splitted1[0].astype('int')
    df['month'] = splitted1[1].astype('int')
    df['day'] = splitted1[2].astype('int')

    df.drop(['date', 'timestamp'], axis=1, inplace=True)

    df['open-close'] = df['open'] - df['close']
    df['low-high'] = df['low'] - df['high']
    df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)

    return df



#---------------------------------------------------------------------------------------------------------

def save_model(model_instance, save_path:str):
    try:
        parsed_save_path = assert_path(save_path)

        from joblib import dump
        with open(parsed_save_path, 'wb') as f:
            print('Saving model instance.')
            dump(model_instance, f)
            print('Model instance successfully saved.')


    except Exception as e:
        raise e
    
#---------------------------------------------------------------------------------------------------------

def load_model(model_path:str):
    try:
        parsed_model_path = assert_path(model_path)

        from joblib import load
        with open(parsed_model_path, 'rb') as f:
            model = load(f)

            return model
        

    except Exception as e:
        raise e
    
#---------------------------------------------------------------------------------------------------------

def train_model(model, x_train, y_train):
    try:
        model.fit(x_train, y_train)
    
        return model
    
    
    except Exception as e:
        raise e
    
#---------------------------------------------------------------------------------------------------------

def test_model(model, x_test, y_test):
    try:
        y_preds = model.predict(x_test)

        from sklearn.metrics import accuracy_score
        score = accuracy_score(y_test, y_preds)

        return score
    

    except Exception as e:
        raise e

#---------------------------------------------------------------------------------------------------------

def get_inference_data():
    pass
