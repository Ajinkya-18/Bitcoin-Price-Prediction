from utils import load_data, preprocess_data, train_model, save_model, test_model

# data loading
df = load_data('data/bitcoin_2017_to_2023.csv')
print(df.shape)

# data preprocessing
x_train, x_test, y_train, y_test = preprocess_data(df, mode='train')
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# model training
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

lr = LogisticRegression(penalty='l2', tol=0.0001, C=0.5, class_weight='balanced', 
                        random_state=42, solver='saga', max_iter=1000, verbose=1, n_jobs=5)

hgbc = HistGradientBoostingClassifier(loss='log_loss', learning_rate=0.1, max_iter=200, 
                                      max_leaf_nodes=31, max_depth=10, min_samples_leaf=100, 
                                      max_features=1.0, max_bins=255, scoring='f1', 
                                      validation_fraction=0.25, n_iter_no_change=10, 
                                      verbose=1, random_state=42, class_weight='balanced')

rfc = RandomForestClassifier(n_estimators=150, max_depth=10, min_samples_split=200, 
                             min_samples_leaf=100, max_features='sqrt', bootstrap=True, 
                             oob_score=True, n_jobs=5, random_state=42, verbose=1, 
                             class_weight='balanced')

lr_trained = train_model(lr, x_train, y_train)
hgbc_trained = train_model(hgbc, x_train, y_train)
rfc_trained = train_model(rfc, x_train, y_train)

# model testing
score = test_model(hgbc_trained, x_test, y_test, scoring='f1')
print(round(float(score)*100, 2))

# saving trained models
save_model(lr_trained, save_path='models/lr_fitted.joblib')

