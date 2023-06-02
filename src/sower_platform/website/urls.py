from django.urls import path
from . import views

urlpatterns = [
    path('management/', views.management_view, name='management'),
]