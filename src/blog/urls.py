from django.urls import path
from .views import index, article, accueil

urlpatterns = [
    path('', index, name="blog-index"),
    path('article-<str:numero_article>/', article, name="blog-article"),
    path('accueil', accueil, name="accueil"),
    
    
]
