

def seq_raw_data(data_path="logistic.npy", val_size = 0.1, test_size = 0.1):
    """ this approach is fundamentally flawed if time series is non-statinary"""
    print("loading sequence data ...")
    data = np.load(data_path)
    if (np.ndim(data)==1):
        data = np.expand_dims(data, axis=1)
    print("input type ",type( data), np.shape(data))

    """normalize the data"""
    print("normalize to (0-1)")
    data = normalize_columns(data)

    ntest = int(round(len(data) * (1 - test_size)))
    nval = int(round(len(data[:ntest]) * (1 - val_size)))
    train_data, valid_data, test_data = data[:nval, ], data[nval:ntest, ], data[ntest:,]
    return train_data, valid_data, test_data
