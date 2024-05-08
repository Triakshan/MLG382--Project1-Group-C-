def prepare():
    # Importing Libraries
    # Suppress warnings
    import warnings
    warnings.simplefilter(action="ignore", category=FutureWarning)

    import numpy as np
    import pandas as pd

    # Load data
    df = pd.read_csv('../data/raw_data.csv')

    # Drop unnecessary column
    df = df.drop('Loan_ID', axis=1)

    # Remove rows with null values
    nullCols = ['Gender', 'Dependents', 'LoanAmount', 'Loan_Amount_Term']
    df = df.dropna(subset=nullCols)

    # Fill categorical values
    df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])

    # Fill numerical values
    # df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])
    df['Credit_History'].fillna(df['Credit_History'].mean(), inplace=True)

    # One Hot Encoding of categorical features
    df['Dependents'] = df['Dependents'].replace(to_replace='3+', value='4')

    df['Gender'] = df['Gender'].map({'Male' : 1, 'Female' : 0}).astype('int')
    df['Married'] = df['Married'].map({'Yes' : 1, 'No' : 0}).astype('int')
    df['Education'] = df['Education'].map({'Graduate' : 1, 'Not Graduate' : 0}).astype('int')
    df['Self_Employed'] = df['Self_Employed'].map({'Yes' : 1, 'No' : 0}).astype('int')
    df['Property_Area'] = df['Property_Area'].map({'Rural' : 0, 'Urban' : 1, 'Semiurban': 2}).astype('int')
    df['Loan_Status'] = df['Loan_Status'].map({'Y' : 1, 'N' : 0}).astype('int')

    # Feature Matrix X and Response Y
    X = df.drop('Loan_Status', axis=1)
    y = df['Loan_Status']

    # Feature Scaling
    cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
    from sklearn.preprocessing import StandardScaler
    ss = StandardScaler()
    X[cols] = ss.fit_transform(X[cols])

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    return X_train, X_test, y_train, y_test