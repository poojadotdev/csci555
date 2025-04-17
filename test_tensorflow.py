import tensorflow as tf
print("TensorFlow version:", tf.__version__)

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(10, activation='relu', input_shape=(5, 1)),
    tf.keras.layers.Dense(1)
])
model.summary()
