from django.shortcuts import render, get_object_or_404
from django.utils import timezone
# from .models import Post
from django.shortcuts import redirect
from django.views.generic.edit import FormView
import nibabel as nib
import numpy as np
from segmentation.plot3d import *
from django.urls import reverse
from django.http import HttpResponse, JsonResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .utils import Unet
import matplotlib.pyplot as plt
from io import BytesIO, StringIO
import io
import urllib
import base64
import cv2
from django.http import HttpResponseRedirect
import pickle
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

from segmentation.unet_v2 import *
from django.views.decorators.csrf import csrf_exempt
from sklearn.preprocessing import MinMaxScaler

# Create your views here.

@csrf_exempt
def predict(request):
    filename = []
    dummy = []
    if request.method == 'POST':
        print("Hello")
        for x in request.FILES.getlist("files"):
            print("Hello")
            print(x.name)
            f = handle_uploaded_file(x)
            # nii_file = nib.load("segmentation/static/upload/"+x.name)
            # dummy.append(nii_file.get_fdata())
            filename.append(f)
            print(f, filename)
        # unet = Unet() #------> Version 1
        # graph = unet.unet_model(dummy[0],dummy[1])
        # print(graph)

        # unet = UNetV2()
        """
        prediction = unet.predict(filename)['Prediction'][0]
        print(type(prediction))
        # print(prediction)
        prediction = (prediction).squeeze().cpu().detach().numpy()
        prediction = np.moveaxis(prediction, (0, 1, 2, 3), (0, 3, 2, 1))
        wt, tc, et = prediction
        print(wt.shape, tc.shape, et.shape)
        prediction = (wt + tc + et)
        prediction = np.clip(prediction, 0, 1)
        print(prediction.shape)
        print(np.unique(prediction))
        og = nib.load('segmentation/static/upload/flair.nii')
        nft_img = nib.Nifti1Image(prediction, og.affine)
        nib.save(nft_img, 'NeuroVision/segmentation/static/upload/predicted'  + '.nii')

        reader = ImageReader('./data', img_size=128, normalize=True, single_class=False)
        viewer = ImageViewer3d(reader, mri_downsample=20)
        fig = viewer.get_3d_scan(0, 't1')
        """
        # return render(request, "segmentation/plot3D.html", context={'fig': fig.to_html()})

        return redirect('/option/')

        # return render(request, 'segmentation/slicedrop/index.html', {'data': dummy[0]})
        # return render(request, 'segmentation/index.html')
        # response = redirect('/segmentation/option.html/')
        # return response
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
    # filename = '3D Animation.html'
    filename = 'Ground_Truth_BraTS20_Training_024_3d.gif'
    # Define the full file path
    filepath = BASE_DIR + '/segmentation/static/gifs/' + filename
    filepath = filepath
    # Open the file for reading content
    path = open(filepath, 'rb')
    # Set the mime type
    mime_type, _ = mimetypes.guess_type(filepath)
    # Set the return value of the HttpResponse
    response = HttpResponse(path, content_type=mime_type)
    # Set the HTTP header for sending to browser
    response['Content-Disposition'] = "attachment; filename=%s" % filename
    # Return the response value
    return response

def options(request) :

    if request.method == 'POST':
        option = request.POST['choice']
        if option == 'animation' :
            return redirect('/plot3D')
        elif option == 'gif' :
            return redirect('/3dgif')
        elif option == 'report' :
            return redirect('/report')
    else : 
        return render(request, "segmentation/option.html")

def gif(request) :
    return render(request, "segmentation/view_gif.html")


def report(request) :
    filename = 'segmentation/static/upload/segmented.nii'
    segmented = getMaskSizesForVolume(nib.load(filename).get_fdata())
    brain_vol = getBrainSizeForVolume(nib.load('segmentation/static/upload/flair.nii').get_fdata())

    segmented[2] = segmented[2]/brain_vol # for edema
    segmented[3] = segmented[3]/brain_vol # for enhancing
    total_tumor_density =  (segmented[1] +  segmented[2] +  segmented[3]) / brain_vol #for whole tumor
    merged=np.array([[segmented[2],segmented[3] ,total_tumor_density]]) # add segments
    # print(type(merged))
    # print(merged)
    pickled_model = pickle.load(open('segmentation/saved_models/linear-v1.sav', 'rb'))
    survival_days = int(pickled_model.predict(merged)[0][0])
    
    print(f'Survival days are : {survival_days}')

    return render(request, "segmentation/report.html", {'Survival_Days' : survival_days})

def getMaskSizesForVolume(image_volume):
    totals = dict([(1, 0), (2, 0), (3, 0)]) # {'1' : 0, '2' : 0,'3' : 0}
    for i in range(100):
        # flatten 2D image into 1D array and convert mask 4 to 3
        arr=image_volume[:,:,i+22].flatten()
        arr[arr == 4] = 3
        
        unique, counts = np.unique(arr, return_counts=True)
        unique = unique.astype(int)
        values_dict=dict(zip(unique, counts))
        for k in range(1,4):
            totals[k] += values_dict.get(k,0)
    return totals

def getBrainSizeForVolume(image_volume):
    total = 0
    for i in range(100):
        arr=image_volume[:,:,i+22].flatten()
        image_count=np.count_nonzero(arr)
        total=total+image_count
    return total