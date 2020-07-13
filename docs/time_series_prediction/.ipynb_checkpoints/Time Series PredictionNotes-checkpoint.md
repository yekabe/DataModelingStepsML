## Used Models

* MODEL 1

```python
    lstm_model.add(LSTM(50, batch_input_shape=(BATCH_SIZE, TIME_STEPS, x_t.shape[2]), dropout=0.0, recurrent_dropout=0.0))
    lstm_model.add(Dropout(0.5))
    lstm_model.add(Dense(20,activation='tanh'))
    lstm_model.add(Dropout(0.5))
    lstm_model.add(Dense(1,activation='linear'))
    opt = keras.optimizers.Adam(learning_rate=0.001)
    lstm_model.compile(loss='mse', optimizer=opt,metrics=['mse'])
```

* MODEL 2

```python
    lstm_model.add(LSTM(50, batch_input_shape=(BATCH_SIZE, TIME_STEPS, x_t.shape[2]), dropout=0.0, recurrent_dropout=0.0))
    lstm_model.add(Dropout(0.5))
    lstm_model.add(Dense(20,activation='tanh'))
    lstm_model.add(Dense(1,activation='relu'))
    opt = keras.optimizers.Adam(learning_rate=0.001)
    lstm_model.compile(loss='mse', optimizer=opt,metrics=['mse'])
```

* MODEL 3

```python
    lstm_model.add(LSTM(50, batch_input_shape=(BATCH_SIZE, TIME_STEPS, x_t.shape[2]), dropout=0.0, recurrent_dropout=0.0))
    lstm_model.add(Dropout(0.5))
    lstm_model.add(Dense(20,activation='relu'))
    lstm_model.add(Dense(1,activation='relu'))
    opt = keras.optimizers.Adam(learning_rate=0.001)
    lstm_model.compile(loss='mse', optimizer=opt,metrics=['mse'])
```


* MODEL 4

```python
    lstm_model.add(LSTM(100, batch_input_shape=(BATCH_SIZE, TIME_STEPS, x_t.shape[2]), dropout=0.0, recurrent_dropout=0.0))
    lstm_model.add(Dropout(0.5))
    lstm_model.add(Dense(20,activation='relu'))
    lstm_model.add(Dense(1,activation='relu'))
    opt = keras.optimizers.Adam(learning_rate=0.001)
    lstm_model.compile(loss='mse', optimizer=opt,metrics=['mse'])
```


* MODEL 5

```python
    lstm_model.add(LSTM(50, batch_input_shape=(BATCH_SIZE, TIME_STEPS, x_t.shape[2]), dropout=0.0, recurrent_dropout=0.0))
    lstm_model.add(Dropout(0.5))
    lstm_model.add(Dense(20,activation='relu'))
    lstm_model.add(Dense(1,activation='relu'))
    # opt = keras.optimizers.Adam(learning_rate=0.001)
    lstm_model.compile(loss='mse', optimizer='rmsprop',metrics=['mse'])
```


* MODEL 6

```python
    lstm_model.add(LSTM(50, batch_input_shape=(BATCH_SIZE, TIME_STEPS, x_t.shape[2]), dropout=0.0, recurrent_dropout=0.0))
    lstm_model.add(Dropout(0.5))
    lstm_model.add(Dense(20,activation='tanh'))
    lstm_model.add(Dense(1,activation='linear'))
    # opt = keras.optimizers.Adam(learning_rate=0.001)
    lstm_model.compile(loss='mse', optimizer='rmsprop',metrics=['mse'])
```


* MODEL 7

```python
    lstm_model.add(LSTM(100, batch_input_shape=(BATCH_SIZE, TIME_STEPS, x_t.shape[2]), dropout=0.0, recurrent_dropout=0.0))
    lstm_model.add(Dropout(0.5))
    lstm_model.add(Dense(20,activation='tanh'))
    lstm_model.add(Dense(1,activation='linear'))
    opt = keras.optimizers.Adam(learning_rate=0.001)
    lstm_model.compile(loss='mse',metrics=['mse'])
```
