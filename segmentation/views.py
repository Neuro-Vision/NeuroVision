from django.shortcuts import render, get_object_or_404
from django.utils import timezone
# from .models import Post
from django.shortcuts import redirect
from django.views.generic.edit import FormView
import nibabel as nib
import numpy as np

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