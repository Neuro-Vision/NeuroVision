from django.urls import path
from . import views

urlpatterns = [
		   	path('', views.predict, name='predict'),
			path('slicedrop', views.slicedrop, name='slicedrop'),
			path('plot3D', views.plot_3D, name='plot3D'),
			path('option/', views.options, name='options'),
   			path('download/', views.download_file, name='download_animation'),
		     ]
