from app.views import EmotionRecognitionView
from django.urls import path

urlpatterns = [
    path("", EmotionRecognitionView.as_view(), name="index"),
]
