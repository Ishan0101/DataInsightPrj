from django.urls import path
from .views import ClearAll, Forecast, Home, MultiVarientExport, RemoveSpecific, SingleInsights, MultiVarientInsights,SingleExport,Export

app_name ='datainsights'

urlpatterns = [
    path('', Home.as_view(), name='home'),
    path('insights/', SingleInsights.as_view(), name='insights'),
    path('multivarientinsights/', MultiVarientInsights.as_view(), name='multivarientinsights'),
    path('forecast/', Forecast.as_view(), name='forecast'),
    path('single_export/', SingleExport.as_view(), name='single_export'),
    path('export_page/', Export.as_view(), name='export_page'),
    path('clear_all/', ClearAll.as_view(), name='clear_all'),
    path('multivarient_export/',MultiVarientExport.as_view(),name='multivarient_export'),
    path('remove_specific/',RemoveSpecific.as_view(),name='remove_specific'),
]