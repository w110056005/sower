from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi

# Login parameter.
@swagger_auto_schema(
    method='get',
    manual_parameters=[
        openapi.Parameter(
            name='node_id',
            in_=openapi.IN_QUERY,
            description='node_id',
            type=openapi.TYPE_STRING
        )
    ]
)
@api_view(['GET'])
def login(request):
    node_id = str(request.query_params.get('node_id', ""))
    return Response(node_id)
