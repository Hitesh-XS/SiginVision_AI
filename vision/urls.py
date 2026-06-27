from django.urls import path
from . import views

urlpatterns = [
    path("collector/", views.collector_page, name="collector_page"),

    path("gestures/", views.gesture_list, name="gesture_list"),
    path("sample-counts/", views.sample_counts, name="sample_counts"),
    path("collect-sample/", views.collect_sample, name="collect_sample"),

    path("predict-static/", views.predict_static, name="predict_static"),
]