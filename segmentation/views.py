from django.shortcuts import render, get_object_or_404
from django.utils import timezone
# from .models import Post
from django.shortcuts import redirect
from django.views.generic.edit import FormView
import nibabel as nib
import numpy as np
from segmentation.plot3d import *

from django.http import HttpResponse, JsonResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .utils import Unet
import matplotlib.pyplot as plt
from io import BytesIO, StringIO
import io
import urllib, base64
import cv2


from .utils import handle_uploaded_file  
from segmentation.forms import UploadFile

import mimetypes
import os
from django.http.response import HttpResponse

from django.shortcuts import render
from plotly.offline import plot
from plotly.graph_objs import Scatter
import pathlib
from django.http import FileResponse


# Create your views here.


def predict(request):
    dummy = []
    if request.method == 'POST':
        for x in request.FILES.getlist("files"):
            handle_uploaded_file(x)
            nii_file = nib.load("segmentation/static/upload/"+x.name)
            dummy.append(nii_file.get_fdata())
        unet = Unet()
        graph = unet.unet_model(dummy[0],dummy[1])
        print(graph)
        
        return render(request, 'segmentation/slicedrop/index.html', {'data':dummy[0]})
    else :
        student = UploadFile()  
        return render(request,"segmentation/index.html")  

def slicedrop(request):
    return render(request, "segmentation/slicedrop/index.html")

def plot_3D(request):
    reader = ImageReader('./data', img_size=128, normalize=True, single_class=False)
    viewer = ImageViewer3d(reader, mri_downsample=20)
    fig = viewer.get_3d_scan(0, 't1')

    return render(request, "segmentation/plot3D.html", context={'fig': fig.to_html()})

def download_file(request):
    # Define Django project base directory
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Define text file name
    filename = '3D Animation.html'
    # Define the full file path
    filepath = BASE_DIR + '/segmentation/' + filename
    # Open the file for reading content
    path = open(filepath, 'r')
    # Set the mime type
    mime_type, _ = mimetypes.guess_type(filepath)
    # Set the return value of the HttpResponse
    response = HttpResponse(path, content_type=mime_type)
    # Set the HTTP header for sending to browser
    response['Content-Disposition'] = "attachment; filename=%s" % filename
    # Return the response value
    return response