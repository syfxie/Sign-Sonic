import os

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.metrics import CategoricalAccuracy

from preprocess import preprocess_data
from data.load_dataset import load_data

data = load_data()

X, y = preprocess_data(data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

frames = 50
output_shape = y_test.shape[1]

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(frames, 1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(output_shape, activation='softmax'))

model.compile(
    optimizer="Adam",
    loss='categorical_crossentropy',
    metrics=[CategoricalAccuracy()]
)

lr = 0.01
epochs = 200
batch_size = 64

log_dir = os.path.join('Logs')
callbacks = TensorBoard(log_dir=log_dir)

history = model.fit(
    x=X_train,
    y=y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose="auto",
    callbacks=callbacks
)

model.summary()

res = model.predict(X_test)

print("Evaluate on test data")
results = model.evaluate(X_test, y_test, batch_size=64)
print("test loss, test acc:", results)
