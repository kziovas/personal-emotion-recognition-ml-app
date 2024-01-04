import os

from app.forms import EmotionRecognitionForm
from app.ml_pipeline import EmotionRecognitionMLPipeline
from app.models import EmotionRecognition
from django.conf import settings
from django.shortcuts import render
from django.urls import reverse_lazy
from django.views.generic.edit import FormView


class EmotionRecognitionView(FormView):
    template_name = "index.html"
    form_class = EmotionRecognitionForm
    success_url = reverse_lazy("index")

    def get(self, request, *args, **kwargs):
        form = self.get_form()
        return render(request, self.template_name, {"form": form, "upload": False})

    def post(self, request, *args, **kwargs):
        form = self.get_form()
        if form.is_valid():
            save = form.save(commit=True)
            primary_key = save.pk
            imgobj = EmotionRecognition.objects.get(pk=primary_key)
            fileroot = str(imgobj.image)
            filepath = os.path.join(settings.MEDIA_ROOT, fileroot)
            results = EmotionRecognitionMLPipeline.analyse_image(filepath)
            return render(
                request,
                self.template_name,
                {"form": form, "upload": True, "results": results},
            )
        else:
            return render(request, self.template_name, {"form": form, "upload": False})
