from django.urls import path
from . import views


urlpatterns = [
    path("gesture/",views.gesture_list),
    path("collect-sample",views.collect_sample),
]
