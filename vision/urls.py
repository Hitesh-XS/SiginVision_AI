from django.urls import path
from . import views

urlpatterns = [
    path("collector/", views.collector_page, name="collector_page"),
    path("predict-page/", views.predict_page, name="predict_page"),
    path("dynamic-predict-page/", views.dynamic_predict_page, name="dynamic_predict_page"),

    path("gestures/", views.gesture_list, name="gesture_list"),
    path("create-gesture/", views.create_gesture, name="create_gesture"),

    path("sample-counts/", views.sample_counts, name="sample_counts"),
    path("dynamic-sample-counts/", views.dynamic_sample_counts, name="dynamic_sample_counts"),

    path("collect-sample/", views.collect_sample, name="collect_sample"),
    path("predict-static/", views.predict_static, name="predict_static"),
    path("predict-dynamic/", views.predict_dynamic, name="predict_dynamic"),
    path("app/",views.signvision_app_page,name="signvision_app_page"),
    path("save-sentence/",views.save_sentence,name="save_sentence"),
]