from app.models import EmotionRecognition
from django import forms


class EmotionRecognitionForm(forms.ModelForm):
    class Meta:
        model = EmotionRecognition
        fields = ["image"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["image"].widget.attrs.update({"class": "form-control"})
