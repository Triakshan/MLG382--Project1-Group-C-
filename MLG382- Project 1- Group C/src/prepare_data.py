def prepareData():
    # Importing the libraries
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # Importing the dataset
    dataset = pd.read_csv('../data/raw_data.csv')

    dataset.dropna(subset=['LoanAmount'], inplace=True)

    X = dataset.iloc[:, 1: -1].values
    y = dataset.iloc[:, -1].values

    # Taking care of missing data
    dataset.isnull().sum()

    # Fill in missing data of all our categorical values
    dataset['Gender'] = dataset['Gender'].fillna(dataset['Gender'].mode()[0])
    dataset['Married']=dataset['Married'].fillna(dataset['Married'].mode()[0])
    dataset['Dependents']=dataset['Dependents'].fillna(dataset['Dependents'].mode()[0])
    dataset['Self_Employed']=dataset['Self_Employed'].fillna(dataset['Self_Employed'].mode()[0])

    # Fill in missing data of all our numerical values
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer.fit(X[:, 7:10])
    X[:, 7:10] = imputer.transform(X[:, 7:10])

    df = pd.DataFrame(X)

    # Encoding categorical data
    # Encoding the Independent Variable
    # Use one hot encoder
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    columns_to_encode = [0, 1, 2, 3, 4, 9, 10]
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), columns_to_encode)], remainder='passthrough')
    X = ct.fit_transform(X)

    # Encoding the Dependent Variable
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    return X_train, X_test, y_train, y_test