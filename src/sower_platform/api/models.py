from django.db import models

# Create your models here.
class Node(models.Model):
    node_name = models.TextField(default="local")
    version = models.TextField(default="latest")
    last_modify_date = models.DateTimeField(auto_now=True)
    created = models.DateTimeField(auto_now_add=True)
    class Meta:
        db_table = "Node"