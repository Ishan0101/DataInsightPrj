from django.urls import path
from .views import Forecast, Home, SingleInsights, MultiVarientInsights

app_name ='datainsights'

urlpatterns = [
    path('', Home.as_view(), name='home'),
    path('insights/', SingleInsights.as_view(), name='insights'),
    path('multivarientinsights/', MultiVarientInsights.as_view(), name='multivarientinsights'),
    path('forecast/', Forecast.as_view(), name='forecast'),
]