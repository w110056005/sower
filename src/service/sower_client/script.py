import zmq
import subprocess
import docker

# Create a ZeroMQ context
context = zmq.Context()
# Create a subscriber socket
socket = context.socket(zmq.SUB)
# Set the subscription filter (empty string means subscribe to all messages)
socket.setsockopt_string(zmq.SUBSCRIBE, '')
# Connect to the publisher's address
socket.connect('tcp://sower_platform_container:5555')

def execute_python_file(file_path):
    try:
        # Execute the Python file as a separate process
        subprocess.run(['python', file_path], check=True)
    except subprocess.CalledProcessError as e:
        # Handle any errors that occur during the execution
        print(f"Error executing {file_path}: {e}")

def update_seed():
    client = docker.from_env()
    client.images.pull('w110056005/seed:latest')

    container = client.containers.get('sower_seed_container')
    container.stop()
    container.remove()

    client.containers.run(
        'w110056005/seed:latest',
        name='sower_seed_container',
        detach=True
    )

def main():
    print("Running Sower client in background...")
    while True:
        message = socket.recv_string()
        print(f'Received message: {message}')
    
        if(message == "Upgrade"):
            update_seed()


if __name__ == "__main__":
    main()




