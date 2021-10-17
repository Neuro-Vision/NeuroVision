from django.urls import path
from . import views

urlpatterns = [
		   	path('', views.predict, name='predict'),
			path('slicedrop', views.slicedrop, name='slicedrop'),
			path('plot3D', views.plot_3D, name='plot3D'),
			path('option/', views.options, name='option'),
   			path('download/', views.download_file, name='download_animation'),
			path('3dgif', views.gif, name = "3dgif"),
			path('survival', views.survival, name = "survival"),
			path('2d_view', views.twoD_view, name = "2d_view")
		     ]
