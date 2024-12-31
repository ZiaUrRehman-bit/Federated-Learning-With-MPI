import socket
import os
import struct

# Define the server address and port
SERVER_HOST = '0.0.0.0'
SERVER_PORT = 5001

# Buffer size for file transmission
BUFFER_SIZE = 4096

# Create a directory to store the received files
os.makedirs('received_files04_06_2024', exist_ok=True)

def receive_file(conn):
    # Receive the filename length
    filename_length = struct.unpack('I', conn.recv(4))[0]

    # Receive the filename
    filename = conn.recv(filename_length).decode()

    # Open the file for writing in binary mode
    with open(os.path.join('received_files04_06_2024', filename), 'wb') as f:
        while True:
            bytes_read = conn.recv(BUFFER_SIZE)
            if not bytes_read:
                break
            f.write(bytes_read)

def main():
    # Create a socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((SERVER_HOST, SERVER_PORT))
    server_socket.listen(5)
    print(f"Server listening on {SERVER_HOST}:{SERVER_PORT}")

    while True:
        conn, addr = server_socket.accept()
        print(f"Connection from {addr} has been established!")

        # Receive the file
        receive_file(conn)

        print(f"File received successfully!")

        # Close the connection
        conn.close()

if __name__ == "__main__":
    main()
