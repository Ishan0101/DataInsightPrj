from django.urls import path
from .views import Home, SingleInsights, MultiVarientInsights

app_name ='datainsights'

urlpatterns = [
    path('', Home.as_view(), name='home'),
    path('insights/', SingleInsights.as_view(), name='insights'),
    path('multivarientinsights/', MultiVarientInsights.as_view(), name='multivarientinsights'),
]