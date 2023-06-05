from django.shortcuts import render
import zmq
from django.shortcuts import render, redirect
from .forms import ManagementForm
from common import util
from shared.zmq_shared import ZmqShared

zmq_shared = ZmqShared()


# Create your views here.
def management_view(request):
    if request.method == 'POST':
        form = ManagementForm(request.POST)
        # Handle form submission
        button = request.POST.get('button')
        # Process the submitted form data and perform actions based on the button clicked
        print("button:"+button)

        if button == 'TrainingStart':
            print("Sending Start...")
            message = "Start"
            zmq_shared.publish_message(message)
            util.execute_python_file("./website/server.py")
            pass
        elif button == 'UpgradeSeed':
            print("Sending Upgrading...")
            message = "Upgrade"
            zmq_shared.publish_message(message)
            pass

    else:
        form = ManagementForm()
    return render(request, 'management.html', {'form': form})

