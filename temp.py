import tensorflow as tf
import time

# Quick test model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1024, activation='relu', input_shape=(2048,)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy'
)

# Random data for testing
x = tf.random.normal([10000, 2048])
y = tf.random.uniform([10000], minval=0, maxval=10, dtype=tf.int64)

start = time.time()
model.fit(x, y, epochs=2, batch_size=128, verbose=1)
end = time.time()

print("Training time:", end - start)
