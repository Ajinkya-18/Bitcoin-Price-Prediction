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
    parsed_data_path = assert_path(data_path)

    if parsed_data_path.endswith('.csv'):
        import pandas as pd
        
        df = pd.read_csv(parsed_data_path)
            
        return df

    else:
        raise ValueError('Invalid file extension !')
    
#---------------------------------------------------------------------------------------------------------

def preprocess_data(df, target:str='target', mode:str='train'):
    import numpy as np
    import pandas as pd

    if isinstance(df, pd.DataFrame):
        splitted = df['timestamp'].str.split(' ', expand=True)
        df['date'] = splitted[0].astype('str')
        df['time'] = splitted[1].astype('str')

        splitted1 = df['date'].str.split('-', expand=True)
        df['year'] = splitted1[0].astype('int')
        df['month'] = splitted1[1].astype('int')
        df['day'] = splitted1[2].astype('int')

        splitted2 = df['time'].str.split(':', expand=True)
        df['hour'] = splitted2[0].astype('int')
        df['minute'] = splitted2[1].astype('int')
        
        df['close-open'] = df['close'] - df['open']
        df['high-low'] = df['high'] - df['low']
        df['wick_length_high'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['wick_length_low'] = df[['open', 'close']].min(axis=1) - df['low']
        df['buy_ratio'] = df['taker_buy_base_asset_volume'] / df['volume']
        df['volume_delta'] = df['volume'] - df['volume'].shift(-1)
        df['trade_activity_rate'] = df['number_of_trades'] / df['volume']

        df.drop(['date', 'time', 'timestamp', 'open', 'high', 'low', 'year', 
                'taker_buy_quote_asset_volume', 'quote_asset_volume', 'number_of_trades'], 
                axis=1, inplace=True)
        
        
        del splitted, splitted1, splitted2
        df = df.astype(np.float32)


        if mode=='train':
            df[target] = np.where(df['close'] > df['close'].shift(-1), 1, 0)
            # Outliers removal
            df = df.drop(df[df['close'] > 62000].index, axis=0)
            df = df.drop(df[df['volume'] > 300].index, axis=0)
            df = df.drop(df[df['taker_buy_base_asset_volume'] > 150].index, axis=0)
            df = df.drop(df[(df['close-open'] > 100) | (df['close-open'] < -100)].index, axis=0)
            df = df.drop(df[(df['high-low'] > 150)].index, axis=0)
            df = df.drop(df[(df['wick_length_high'] > 50)].index, axis=0)
            df = df.drop(df[(df['wick_length_low'] > 50)].index, axis=0)
            df = df.drop(df[(df['volume_delta'] < -250) | (df['volume_delta'] > 250)].index, axis=0)
            df = df.drop(df[df['trade_activity_rate'] > 150].index, axis=0)

            df.dropna(inplace=True)

            x_train, x_test, y_train, y_test = split_data(df, target=target)

            rfecv = load_model('models/rfecv_rfc.joblib')
            scaler = load_model('models/rob_scaler.joblib')

            x_train_new = rfecv.transform(x_train)
            x_test_new = rfecv.transform(x_test)

            x_train_new_scaled = scaler.transform(x_train_new)
            x_test_new_scaled = scaler.transform(x_test_new)

            return x_train_new_scaled, x_test_new_scaled, y_train, y_test
        

        if mode=='inference':
            df.dropna(inplace=True)
            # print(df.shape)
            rfecv = load_model('models/rfecv_rfc.joblib')
            scaler = load_model('models/rob_scaler.joblib')

            df_new = rfecv.transform(df)
            df_new_scaled = scaler.transform(df_new)

            return df_new_scaled
        
    else:
        raise TypeError('Input "df" must be a pandas DataFrame.')

#---------------------------------------------------------------------------------------------------------

def split_data(df, target:str):
    import pandas as pd
    try:
        if isinstance(df, pd.DataFrame):
            X, Y = df.drop([target], axis=1), df[target]

            from sklearn.model_selection import train_test_split
            x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

            return x_train, x_test, y_train, y_test
        
        else:
            raise TypeError("Input 'df' must be a pandas DataFrame.")
    

    except Exception as e:
        raise RuntimeError(f'Error occurred during data split: {e}')
    
#----------------------------------------------------------------------------------------------------------

def save_model(model_instance, save_path:str):
    import os

    try:
        if save_path.endswith('.joblib'):
            cwd = os.getcwd()
            parsed_save_path = os.path.join(cwd, save_path)

            from joblib import dump
            with open(parsed_save_path, 'wb') as f:
                print('Saving model instance.')
                dump(model_instance, f)
                print('Model instance successfully saved.')


    except Exception as e:
        raise RuntimeError(f'Error occurred during model saving: {e}')
    
#---------------------------------------------------------------------------------------------------------

def load_model(model_path:str):
    try:
        parsed_model_path = assert_path(model_path)

        from joblib import load
        with open(parsed_model_path, 'rb') as f:
            model = load(f)

            return model
        

    except Exception as e:
        raise RuntimeError(f'Error occurred during model loading: {e}')
    
#---------------------------------------------------------------------------------------------------------

def train_model(model, x_train, y_train):
    try:
        model.fit(x_train, y_train)
    
        return model
    
    
    except Exception as e:
        raise RuntimeError(f'Error occurred during model training: {e}')
    
#---------------------------------------------------------------------------------------------------------

def test_model(model, x_test, y_test, scoring:str='accuracy'):
    try:
        y_preds = model.predict(x_test)

        if scoring == 'accuracy':
            from sklearn.metrics import accuracy_score
            score = accuracy_score(y_test, y_preds)

            return score
        
        if scoring == 'f1':
            from sklearn.metrics import f1_score
            score = f1_score(y_test, y_preds)

            return score
    
        else:
            raise ValueError('"scoring" only takes "accuracy" and "f1" as valid input values.')
    except Exception as e:
        raise RuntimeError(f'Error occurred during model testing: {e}')

#---------------------------------------------------------------------------------------------------------

def get_inference_data():
    from datetime import datetime
    from random import choice, uniform
    import pandas as pd

    inf_data = {'timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S'), datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                'open': [uniform(2830.0, 69000.0), uniform(2830.0, 69000.0)],
                'high': [uniform(2830.0, 69000.0), uniform(2830.0, 69000.0)],
                'low': [uniform(2817.0, 68786.7), uniform(2817.0, 68786.7)],
                'close': [uniform(2817.0, 69000.0), uniform(2817.0, 69000.0)],
                'volume': [uniform(0.0, 5877.77), uniform(0.0, 5877.77)],
                'quote_asset_volume': [uniform(0.0, 145955668.33), uniform(0.0, 145955668.33)],
                'number_of_trades': [choice(range(0, 107315)), choice(range(0, 107315))],
                'taker_buy_base_asset_volume': [uniform(0.0, 3537.45), uniform(0.0, 3537.45)],
                'taker_buy_quote_asset_volume': [uniform(0.0, 89475505.033), uniform(0.0, 89475505.033)]
                }
    
    inf_df = pd.DataFrame.from_dict(inf_data)

    return inf_df

#------------------------------------------------------------------------------------------------------------




