from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response

# Create your views here.
@api_view(['GET'])
def item_list(request):
    items = ['item1', 'item2', 'item3']
    return Response(items)
