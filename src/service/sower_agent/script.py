import docker
import random

from paho.mqtt import client as mqtt_client

broker = 'broker.emqx.io'
port = 1883
topic = "Sower"
# Generate a Client ID with the subscribe prefix.
client_id = f'subscribe-{random.randint(0, 100)}'
# username = 'emqx'
# password = 'public'

def connect_mqtt() -> mqtt_client:
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)

    client = mqtt_client.Client(mqtt_client.CallbackAPIVersion.VERSION1, client_id)
    # client.username_pw_set(username, password)
    client.on_connect = on_connect
    client.connect(broker, port)
    return client

def subscribe(client: mqtt_client):
    def on_message(client, userdata, msg):
        print('Receive msg: '+ msg.payload.decode())
        if("Upgrade" in msg.payload.decode()):
            version  = msg.payload.decode().split(',')[1]
            update_seed(version)
    client.subscribe(topic)
    client.on_message = on_message

def update_seed(version):
    print('Enter update_seed()')
    client = docker.from_env()
    client.images.pull('w110056005/seed:' + version)
    print('Image pull completed.')
    try:
        print('Removing legacy container.')
        container = client.containers.get('sower_seed_container')
        container.stop()
        container.remove()
        print('Legacy Stopped removed.')
        print('Seed is running.')
    except:
        print("no running sower_seed_container...")

    print('starting seed...')
    client.containers.run(
        'w110056005/seed:'+ version,
        name='sower_seed_container',
        detach=True, 
        links={'sower_platform_container': 'sower_platform_container'},  # Link to server container
    )
    print('seed started!')

def start():
    print("Running Sower Agent in background...")
    client = connect_mqtt()
    subscribe(client)
    client.loop_forever()

def login():
    import requests
    r = requests.get('http://sower_platform_container:8000/login', params={'node_id' : node_id})
    print(r.text+ ' registered.')

def main():
    import socket
    import sys
    
    if len(sys.argv) >= 2:
        action = sys.argv[1]
    if len(sys.argv) == 3:
        arg = sys.argv[2]


    if action == "start":
        start()
    elif action == "login":
        global node_id
        if arg  is not None:
            node_id = arg
        else:
            node_id = socket.gethostname()
        login()

if __name__ == "__main__":
    main()




