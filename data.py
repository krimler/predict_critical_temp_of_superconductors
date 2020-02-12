def read_atomic_data(path):
    if not path or not os.path.exists(path) or not os.path.isfile(path):
        print("To begin with, your path to data should be proper!")
        sys.exit(1)
    df = pd.read_csv(path)
    columns = df.columns.tolist() # get the columns
    columns = columns[:-1]
    df = pd.read_csv(path, usecols=columns)
    return df, columns

def get_dataset(df, columns):
    X = df[col[:-1]]
    y = df.critical_temp
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    return (X_train, X_test, y_train, y_test)

