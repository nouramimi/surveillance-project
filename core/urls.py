from django.urls import path
from .views import run_algos_and_download,get_teacher_schedule

urlpatterns = [
    path('run-algos/', run_algos_and_download, name='run_algos'),
    path('surveillances/<str:nom_enseignant>/', get_teacher_schedule, name='get_teacher_schedule'),
]
