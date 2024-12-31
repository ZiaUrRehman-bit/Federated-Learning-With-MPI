import numpy as np
from mpi4py import MPI
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import pickle
import socket
import os
import struct

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def build_model():
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
    return model

def train_model(model, x_train, y_train):
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2, verbose=0)

def save_and_send_files(model, test_acc, test_loss, training_time, report, conf_matrix, machine_name, server_ip):
    # Save the metrics and model files
    metrics_file = f'metrics_machine{machine_name}.txt'
    model_json = model.to_json()
    with open(f'model{machine_name}_architecture.json', 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(f'model{machine_name}_weights.weights.h5')
    with open(f'model{machine_name}_paths.pkl', 'wb') as f:
        pickle.dump({'model_architecture': f'model{machine_name}_architecture.json', 'model_weights': f'model{machine_name}_weights.weights.h5'}, f)

    # Save classification metrics
    with open(metrics_file, 'w') as f:
        f.write(f"Test accuracy: {test_acc}\nTest Loss: {test_loss}\n")
        f.write(f"Total Training Time: {training_time}\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(conf_matrix))
    
    # Define server address and port
    SERVER_HOST = server_ip
    SERVER_PORT = 5001
    BUFFER_SIZE = 4096

    files_to_send = [f'model{machine_name}_architecture.json', f'model{machine_name}_weights.weights.h5', f'model{machine_name}_paths.pkl', metrics_file]

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

    for filename in files_to_send:
        if os.path.exists(filename):
            print(f"Sending {filename}...")
            send_file(filename)
            print(f"{filename} sent successfully!")
        else:
            print(f"{filename} does not exist!")

def main():
    # Load and preprocess the dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.expand_dims(x_train, axis=-1) / 255.0  # Normalize pixel values
    x_test = np.expand_dims(x_test, axis=-1) / 255.0
    y_train = to_categorical(y_train, num_classes=10)  # One-hot encode labels
    y_test = to_categorical(y_test, num_classes=10)

    # Distribute data among MPI processes (optional)
    chunk_size = len(x_train) // size
    start_idx = rank * chunk_size
    end_idx = (rank + 1) * chunk_size if rank < size - 1 else len(x_train)
    x_train_chunk = x_train[start_idx:end_idx]
    y_train_chunk = y_train[start_idx:end_idx]

    # Build and train the local model
    model = build_model()
    start_time = time.time()
    train_model(model, x_train_chunk, y_train_chunk)
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total Time Taken by process {rank} is {total_time}")
    
    gathered_time = comm.gather(total_time, root=0)
    # Gather all models to process 0
    gathered_models = comm.gather(model, root=0)

    if rank == 0:
        # Predictions of all models on the test dataset
        print(f"Total time: {gathered_time}")
        print(f"Average Time: {sum(gathered_time)/size}")
        all_predictions = []
        for gathered_model in gathered_models:
            predictions = np.argmax(gathered_model.predict(x_test), axis=1)
            all_predictions.append(predictions)

        # Ensemble prediction by majority voting
        ensemble_predictions = np.stack(all_predictions, axis=0)
        ensemble_predictions = np.transpose(ensemble_predictions)
        final_predictions = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=ensemble_predictions)

        # Calculate ensemble accuracy
        ensemble_accuracy = np.mean(final_predictions == np.argmax(y_test, axis=1))
        print(f"Ensemble Model Test Accuracy: {ensemble_accuracy}")

        y_test_int = np.argmax(y_test, axis=1)

        # Calculate precision, recall, F1-score, and accuracy
        precision = precision_score(y_test_int, final_predictions, average='weighted')
        recall = recall_score(y_test_int, final_predictions, average='weighted')
        f1 = f1_score(y_test_int, final_predictions, average='weighted')
        accuracy = accuracy_score(y_test_int, final_predictions)

        # Print the results
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1-score: {f1}')
        print(f'Accuracy: {accuracy}')

        # Generate classification report and confusion matrix
        report = classification_report(y_test_int, final_predictions, digits=4)
        conf_matrix = confusion_matrix(y_test_int, final_predictions)

        # Prompt for server details
        server_ip = input("Enter Server IP: ")
        machine_name = input("Enter Machine Name: ")

        # Save and send files
        save_and_send_files(model, accuracy, ensemble_accuracy, total_time, report, conf_matrix, machine_name, server_ip)

if __name__ == "__main__":
    main()
