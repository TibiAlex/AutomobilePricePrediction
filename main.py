import json

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def encodestring(df):
    df['marca'] = label_encoder.fit_transform(df['marca'])
    df['model'] = label_encoder.fit_transform(df['model'])
    df['cutie_de_viteze'] = label_encoder.fit_transform(df['cutie_de_viteze'])
    df['combustibil'] = label_encoder.fit_transform(df['combustibil'])
    df['transmisie'] = label_encoder.fit_transform(df['transmisie'])
    df['caroserie'] = label_encoder.fit_transform(df['caroserie'])
    df['culoare'] = label_encoder.fit_transform(df['culoare'])
    df['optiuni_culoare'] = label_encoder.fit_transform(df['optiuni_culoare'])


if __name__ == '__main__':
    # read the values from csv files
    train_path = './output_train.csv'
    test_path = "output_test.csv"
    train_df = pd.read_csv(train_path, encoding='latin-1')
    test_df = pd.read_csv(test_path, encoding='latin-1')

    # drop all nan from train and price column from test
    train_df = train_df.dropna()
    test_df = test_df.drop('pret', axis=1)

    # see the number of unique values in each feature
    # unique_counts = train_df.nunique()
    # print(unique_counts)

    # encode the string values
    label_encoder = LabelEncoder()
    encodestring(train_df)
    encodestring(test_df)

    # pd.set_option('display.max_rows', None)
    # pd.set_option('display.max_columns', None)

    # select the features
    features = [
        'marca',
        'model',
        'an',
        'km',
        'putere',
        'capacitate_cilindrica',
        # 'addons',
        'cutie_de_viteze',
        'combustibil',
        'transmisie',
        'caroserie',
        # 'culoare',
        # 'optiuni_culoare'
    ]

    # scale the feature values
    scaler = StandardScaler()
    X = scaler.fit_transform(train_df[features])
    y = train_df['pret']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # declare the random forest model and train
    random_forest_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
        criterion='entropy',
        bootstrap=True,
        random_state=42
    )
    random_forest_model.fit(X_train, y_train)
    # Make predictions on the test set
    y_pred = random_forest_model.predict(X_test)
    # Evaluate the model
    r2 = r2_score(y_test, y_pred)
    print(r2)

    # Predict the prices
    X_test_scaled = scaler.transform(test_df[features])
    y_pred_test = random_forest_model.predict(X_test_scaled)
    print(y_pred_test)

    # write data inside a json file
    data = None
    json_path = "test.json"
    with open(json_path, 'r') as json_file:
        data = json.load(json_file)
        for i in range(len(y_pred_test)):
            data[i]['pret'] = y_pred_test[i]

    with open("test_result.json", 'w') as json_file:
        json.dump(data, json_file, indent=2)
