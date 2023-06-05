import zmq

class ZmqShared:
    def publish_message(self, message):
        with zmq.Context() as ctx:
            with ctx.socket(zmq.PUB) as s:
                s.bind("tcp://*:5555")
                s.send_string(message)
            # exiting Socket context closes socket
        # exiting Context context terminates context