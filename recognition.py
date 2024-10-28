import tensorflow as tf
import numpy as np
import matplotlib as plt

(x_train, y_train),(x_test, y_test) = tf.keras.datasets.mnist.load_data()

print('Training data shape:', x_train.shape)
print('Testing data shape:', x_test.shape)

print('Shape of a single image:', x_train[0].shape)


# Normalize the data
x_train = x_train/255.0
x_test = x_test/255.0

# Reshape the data
x_train = x_train.reshape(-1, 28 * 28)
x_test = x_test.reshape(-1, 28 * 28)


#Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(784,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])


#Model selection
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# Training the model
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Test loss: {test_loss:.4f}")

# Get a random test image
index = np.random.randint(0, len(x_test))
sample_image = x_test[index]

# Reshapes back to the original shape (28x28)
sample_image_reshaped = sample_image.reshape(28,28)

#Make a prediction
predicted_probabilities = model.predict(sample_image.reshape(1, 784))
predicted_label = np.argmax(predicted_probabilities)

print(f'Predicted label: {predicted_label}')
print(f'Actual label: {y_test[index]}')

# Save the model
model.save("mnist_digit_recognition_model.h5")
