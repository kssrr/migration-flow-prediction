# migration-flow-prediction

Predicting bilateral migration flows using LSTM-RNNs

## TODO

* One-hot encoding for either dyads or origin and destination
    - Would be great if we could do this already for the model-based imputation, but unfortunately drastically increases the computation time there.
* Non-random train-test split:
    - Look into predicting last year in series for testing, second to last for validation, and all earlier years for training.
* Experiment with different architectures and sequence lengths
* Build proper competitors to benchmark against (conventional time series models, MLP, etc.).
