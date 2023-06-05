from django.shortcuts import render
from rest_framework.decorators import api_view
from common import util
from django.http import HttpResponse
from shared.zmq_shared import ZmqShared

zmq_shared = ZmqShared()

# Create your views here.
@api_view(['GET'])
def training_trigger(request):
    print("Sending Start...")
    message = "Start"
    zmq_shared.publish_message(message)
    util.execute_python_file("./website/server.py")
    return HttpResponse()

@api_view(['GET'])
def updating_trigger(request):
    print("Sending Update...")
    message = "Update"
    zmq_shared.publish_message(message)
    return HttpResponse()

