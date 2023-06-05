from django.urls import path
from . import views

urlpatterns = [
    path('trigger/trian', views.training_trigger, name='training-trigger'),
    path('trigger/update', views.updating_trigger, name='updating-trigger'),

]