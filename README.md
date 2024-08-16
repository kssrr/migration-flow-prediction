# migration-flow-prediction

Predicting bilateral migration flows using LSTM-RNNs

## TODO

* Find better/more data
    - Maybe not only OECD countries? Problem here is that whole time series of features are missing for many non-OECD countries and have to be entirely imputed.
    - Dyadic data would be good/interesting
* Improve LSTM
    - Does not perform better than conventional models (`benchmarks.ipynb`)
* Non-random train-test split
    - Predict last years in the series as test set (forecasting)
