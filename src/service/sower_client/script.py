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

    client = mqtt_client.Client(client_id)
    # client.username_pw_set(username, password)
    client.on_connect = on_connect
    client.connect(broker, port)
    return client

def subscribe(client: mqtt_client):
    def on_message(client, userdata, msg):
        print('Receive msg: '+ msg.payload.decode())
        if(msg.payload.decode() == "Upgrade"):
            update_seed()
    client.subscribe(topic)
    client.on_message = on_message

def update_seed():
    print('Enter update_seed()')
    client = docker.from_env()
    client.images.pull('w110056005/seed:cifar10-latest')
    print('Image pull completed.')
    try:
        print('Removing legacy container.')
        container = client.containers.get('sower_seed_container')
        container.stop()
        container.remove()
        print('Legacy comtainer removed.')
        print('Seed is running.')
    except:
        print("no running sower_seed_container...")

    print('starting seed.')
    client.containers.run(
        'w110056005/seed:cifar10-latest',
        name='sower_seed_container',
        detach=True, 
        links={'sower_platform_container': 'sower_platform_container'},  # Link to server container
    )

def main():
    print("Running Sower client in background...")
    client = connect_mqtt()
    subscribe(client)
    client.loop_forever()

if __name__ == "__main__":
    main()




