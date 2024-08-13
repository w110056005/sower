from django.shortcuts import render
from paho.mqtt import client as mqtt_client
import subprocess
from django.shortcuts import render, redirect
from .forms import ManagementForm
import random
import time

broker = 'broker.emqx.io'
port = 1883
topic = "Sower"
# Generate a Client ID with the publish prefix.
client_id = f'publish-{random.randint(0, 1000)}'
# username = 'emqx'
# password = 'public'

def connect_mqtt():
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)

    client = mqtt_client.Client(mqtt_client.CallbackAPIVersion.VERSION1,client_id)
    # client.username_pw_set(username, password)
    client.on_connect = on_connect
    client.connect(broker, port)
    return client


def publish(client, msg):
    time.sleep(1)
    result = client.publish(topic, msg)
    # result: [0, 1]
    status = result[0]
    if status == 0:
        print(f"Send `{msg}` to topic `{topic}`")
    else:
        print(f"Failed to send message to topic {topic}")


def execute_python_file(file_path, port):
    try:
        # Execute the Python file as a separate process
        subprocess.run(['python', file_path, port], check=True)
    except subprocess.CalledProcessError as e:
        # Handle any errors that occur during the execution
        print(f"Error executing {file_path}: {e}")

# Create your views here.
def management_view(request):
    client = connect_mqtt()
    if request.method == 'POST':
        form = ManagementForm(request.POST)
        # Handle form submission
        button = request.POST.get('button')
        # Process the submitted form data and perform actions based on the button clicked
        print("button:"+button)
        if button == 'TrainingStart':
            print("Sending Start...")
            port = "8090"
            message = "Start,"+port
            print("Send: {message}")
            publish(client, message)
            execute_python_file("./website/server.py", port)
            pass
        elif button == 'UpgradeSeed':
            print("Sending Upgrading...")
            message = "Upgrade"
            publish(client, message)
            pass

    else:
        form = ManagementForm()
    return render(request, 'management.html', {'form': form})

