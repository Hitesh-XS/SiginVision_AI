from django.urls import path
from . import views

urlpatterns = [
    path("gestures/", views.gesture_list, name="gesture_list"),
    path("collect-sample/", views.collect_sample, name="collect_sample"),
    path('collector/',views.collector_page, name="collector_page"),
    path('sample-counts/',views.sample_counts,name="sample_counts"),
]