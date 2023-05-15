import zmq

context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect("tcp://localhost:5555")
socket.setsockopt(zmq.SUBSCRIBE, b"")

print("Running Sower service in background...")

while True:
    message = socket.recv_string()
    print("Received message: %s" % message)