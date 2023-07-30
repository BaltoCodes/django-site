"""DocBlog URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include

from .views import index, accueil, graph_view, calculator_view, get_message, world_is_yours, new_world, human

urlpatterns = [
    path('', index, name="index"),
    path('blog/', include("blog.urls")),
    path('admin/', admin.site.urls),
    path('prix/', accueil, name="accueil"),
    path('graph/', graph_view, name="graph"),
    path('calculator/', calculator_view, name="calculator_view"),
    path('get-message/', get_message, name="get_message"),
    path('accueil/', world_is_yours, name="world"),
    path('new_accueil/', new_world, name="new_earth" ),
    path('human/', human, name="human" ),
]
