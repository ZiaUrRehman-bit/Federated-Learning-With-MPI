import numpy as np
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import socket
import os
import struct

serverIP = input("Enter Server IP: ")

machineName = int(input("Enter Machine Name:"))

# Load and preprocess the dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.expand_dims(x_train, axis=-1) / 255.0  # Normalize pixel values
x_test = np.expand_dims(x_test, axis=-1) / 255.0
y_train = to_categorical(y_train, num_classes=10)  # One-hot encode labels
y_test = to_categorical(y_test, num_classes=10)

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

stTime = time.time()
# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2)
endTime = time.time()
training_time = endTime - stTime

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f'Test accuracy: {test_acc}\tTest Loss: {test_loss}')
print(f"Total Training Time: {training_time}")

# Predict the classes
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Generate classification report
report = classification_report(y_true, y_pred_classes, digits=4)
conf_matrix = confusion_matrix(y_true, y_pred_classes)

# Save the metrics and confusion matrix to a text file
metrics_file = f'metrics_machine{machineName}.txt'
with open(metrics_file, 'w') as f:
    f.write(f"Test accuracy: {test_acc}\nTest Loss: {test_loss}\n")
    f.write(f"Total Training Time: {training_time}\n")
    f.write("Classification Report:\n")
    f.write(report)
    f.write("\nConfusion Matrix:\n")
    f.write(np.array2string(conf_matrix))

# Save the model architecture to JSON
model_json = model.to_json()
with open(f'model{machineName}_architecture.json', 'w') as json_file:
    json_file.write(model_json)

# Save the model weights to a .weights.h5 file
model.save_weights(f'model{machineName}_weights.weights.h5')

# Save the architecture and weights file paths using pickle
with open(f'model{machineName}_paths.pkl', 'wb') as f:
    pickle.dump({'model_architecture': f'model{machineName}_architecture.json', 'model_weights': f'model{machineName}_weights.weights.h5'}, f)

# Define the server address and port
SERVER_HOST = serverIP # Change to the server's IP address if needed
SERVER_PORT = 5001
BUFFER_SIZE = 4096

# Define the file paths to be sent
files_to_send = [f'model{machineName}_architecture.json', f'model{machineName}_weights.weights.h5', f'model{machineName}_paths.pkl', metrics_file]

def send_file(filename):
    # Create a socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((SERVER_HOST, SERVER_PORT))

    # Send the filename length and filename
    filename_bytes = filename.encode()
    client_socket.send(struct.pack('I', len(filename_bytes)))
    client_socket.send(filename_bytes)

    # Send the file content
    with open(filename, 'rb') as f:
        while True:
            bytes_read = f.read(BUFFER_SIZE)
            if not bytes_read:
                break
            client_socket.sendall(bytes_read)

    # Close the connection
    client_socket.close()

def main():
    for filename in files_to_send:
        if os.path.exists(filename):
            print(f"Sending {filename}...")
            send_file(filename)
            print(f"{filename} sent successfully!")
        else:
            print(f"{filename} does not exist!")

if __name__ == "__main__":
    main()
