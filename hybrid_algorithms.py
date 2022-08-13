from keras.layers import LSTM, Dense, Dropout, TimeDistributed, Flatten
from keras.models import Sequential


class LSTMModel:
    def __init__(self, n_lstm_nodes, n_dense_nodes, dropout_rate, activation='tanh'):  #
        self.kwargs = None
        self.base_model = Sequential()

        # Add LSTM Model
        if dropout_rate != 0:
            self.base_model.add(Dropout(dropout_rate))
        self.base_model.add(LSTM(n_lstm_nodes, activation=activation))
        # Use Dropout layer to reduce overfitting of the model to the training data
        # Add one hidden layer with 100 default nodes with relu activation function
        self.base_model.add(Dense(n_dense_nodes, activation=activation))
        # Add output layer with 6 (Total number of classes) nodes with softmax activation function
        self.base_model.add(Dense(1))
        # Compile model
        self.base_model.compile(loss='mae', optimizer='adam')

    def fit(self, x_train, y_train, epochs=100, batch_size=24, verbose=1):
        if len(x_train.shape) != 3:
            import hybrid_preprocess
            x_train = \
                hybrid_preprocess.reshape_features(x_train,
                                                   reshape_features_method=hybrid_preprocess.ReshapeMethod.ThreeDShape)
        return self.base_model.fit(x=x_train, y=y_train, epochs=epochs, shuffle=False, batch_size=batch_size,
                                   verbose=verbose,
                                   use_multiprocessing=True)

    def predict(self, x_test, batch_size=24):
        if len(x_test.shape) != 3:
            import hybrid_preprocess
            x_test = \
                hybrid_preprocess.reshape_features(x_test,
                                                   reshape_features_method=hybrid_preprocess.ReshapeMethod.ThreeDShape)
        result = self.base_model.predict(x=x_test, batch_size=batch_size)
        return result.reshape(-1)

    def set_params(self, **params):
        """Set the parameters of this estimator.  Modification of the sklearn method to
        allow unknown kwargs. This allows using the full range of xgboost
        parameters that are not defined as member variables in sklearn grid
        search.

        Returns
        -------
        self

        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self

        # this concatenates kwargs into parameters, enabling `get_params` for
        # obtaining parameters from keyword parameters.
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.kwargs[key] = value

        return self
