from django.urls import path
from . import views
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path('cvd/', views.predict, name='cvd'),
    path('result/', views.risk_gauge_chart, name='rst'),
    # path("eye_prediction/", views.eye_prediction, name="eye_prediction")
    path("res/", views.upl, name='upl')
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
