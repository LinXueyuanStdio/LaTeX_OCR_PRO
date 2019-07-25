from django.conf.urls import url
from django.contrib import admin
from . import run_model

urlpatterns = [
    url(r'^search-form$', run_model.search_form),
    url(r'^search$', run_model.search),
]
