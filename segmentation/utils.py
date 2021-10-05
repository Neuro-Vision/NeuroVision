import os
import cv2
import glob
import PIL
import shutil
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import plotly
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import albumentations as A
from skimage import data
from skimage.util import montage 
import skimage.transform as skTrans
from skimage.transform import rotate
from skimage.transform import resize
from PIL import Image, ImageOps 
 

# neural imaging
import nilearn as nl
import nibabel as nib
import nilearn.plotting as nlplt
# !pip install git+https://github.com/miykael/gif_your_nifti # nifti to gif 
# import gif_your_nifti.core as gif2nif


# ml libs
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import CSVLogger
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.layers.experimental import preprocessing
from io import StringIO
import io
import urllib, base64
from io import BytesIO

class Unet():
    def __init__(self):

        self.IMG_SIZE=128
        # Make numpy printouts easier to read.
        np.set_printoptions(precision=3, suppress=True)

        # DEFINE seg-areas  
        self.SEGMENT_CLASSES = {
                0 : 'Normal',  #(normal)
                1 : 'Enhancing', # or NON-ENHANCING tumor CORE (enhancing)
                2 : 'Edema',
                3 : 'Non-Enhancing' # original 4 -> converted into 3 later (non-enhancing)
            }

        # there are 155 slices per volume
        # to start at 5 and use 145 slices means we will skip the first 5 and last 5 
        self.VOLUME_SLICES = 100 
        self.VOLUME_START_AT = 22 # first slice of volume that we will include
        self.model = keras.models.load_model('segmentation/model/model_x1_1.h5', 
                                        custom_objects={ 'accuracy' : tf.keras.metrics.MeanIoU(num_classes=4),
                                                        "dice_coef": self.dice_coef,
                                                        "precision": self.precision,
                                                        "sensitivity":self.sensitivity,
                                                        "specificity":self.specificity,
                                                        "dice_coef_enhancing": self.dice_coef_enhancing,
                                                        "dice_coef_edema": self.dice_coef_edema,
                                                        "dice_coef_non_enhancing": self.dice_coef_non_enhancing
                                                        }, compile=False)
        
    name = "predict"
    # dice loss as defined above for 4 classes
    def dice_coef(self, y_true, y_pred, smooth=1.0):
        class_num = 4
        for i in range(class_num):
            y_true_f = K.flatten(y_true[:,:,:,i])
            y_pred_f = K.flatten(y_pred[:,:,:,i])
            intersection = K.sum(y_true_f * y_pred_f)
            loss = ((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))
    #     K.print_tensor(loss, message='loss value for class {} : '.format(SEGMENT_CLASSES[i]))
            if i == 0:
                total_loss = loss
            else:
                total_loss = total_loss + loss
        total_loss = total_loss / class_num
    #    K.print_tensor(total_loss, message=' total dice coef: ')
        return total_loss


    def dice_coef_loss(self, y_true, y_pred):
        return 1-self.dice_coef(y_true, y_pred)

    
    # define per class evaluation of dice coef
    # inspired by https://github.com/keras-team/keras/issues/9395
    def dice_coef_enhancing(self, y_true, y_pred, epsilon=1e-6): #Enhancing
        intersection = K.sum(K.abs(y_true[:,:,:,1] * y_pred[:,:,:,1]))
        return (2. * intersection) / (K.sum(K.square(y_true[:,:,:,1])) + K.sum(K.square(y_pred[:,:,:,1])) + epsilon)

    def dice_coef_edema(self, y_true, y_pred, epsilon=1e-6): # Edema
        intersection = K.sum(K.abs(y_true[:,:,:,2] * y_pred[:,:,:,2]))
        return (2. * intersection) / (K.sum(K.square(y_true[:,:,:,2])) + K.sum(K.square(y_pred[:,:,:,2])) + epsilon)

    def dice_coef_non_enhancing(self, y_true, y_pred, epsilon=1e-6): # Non-Enhancing
        intersection = K.sum(K.abs(y_true[:,:,:,3] * y_pred[:,:,:,3]))
        return (2. * intersection) / (K.sum(K.square(y_true[:,:,:,3])) + K.sum(K.square(y_pred[:,:,:,3])) + epsilon)


    # Ref: salehi17, "Twersky loss function for image segmentation using 3D FCDN"
    # -> the score is computed for each class separately and then summed
    # alpha=beta=0.5 : dice coefficient
    # alpha=beta=1   : tanimoto coefficient (also known as jaccard)
    # alpha+beta=1   : produces set of F*-scores
    # implemented by E. Moebel, 06/04/18
    def tversky_loss(self, y_true, y_pred):
        alpha = 0.5
        beta  = 0.5
        
        ones = K.ones(K.shape(y_true))
        p0 = y_pred      # proba that voxels are class i
        p1 = ones-y_pred # proba that voxels are not class i
        g0 = y_true
        g1 = ones-y_true
        
        num = K.sum(p0*g0, (0,1,2))
        den = num + alpha*K.sum(p0*g1,(0,1,2)) + beta*K.sum(p1*g0,(0,1,2))
        
        T = K.sum(num/den) # when summing over classes, T has dynamic range [0 Ncl]
        
        Ncl = K.cast(K.shape(y_true)[-1], 'float32')
        return Ncl-T


    #intersection over union
    def iou(self, y_true, y_pred, smooth = 0.5):
        intersection = K.sum(y_true * y_pred)
        sum_ = K.sum(y_true + y_pred)
        jac = (intersection + smooth) / (sum_ - intersection + smooth)
        return jac


    # jaccard's loss
    def jac_distance(self, y_true, y_pred):
        y_truef=K.flatten(y_true)
        y_predf=K.flatten(y_pred)

        return - self.iou(y_true, y_pred)


    # Computing Precision 
    def precision(self, y_true, y_pred):
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
            precision = true_positives / (predicted_positives + K.epsilon())
            return precision

        
    # Computing Sensitivity      
    def sensitivity(self, y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        return true_positives / (possible_positives + K.epsilon())


    # Computing Specificity
    def specificity(self, y_true, y_pred):
        true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
        possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
        return true_negatives / (possible_negatives + K.epsilon())


    def predictByPath(self,flair_data,t1ce_data):
    
        X = np.empty((self.VOLUME_SLICES, self.IMG_SIZE, self.IMG_SIZE, 2))

        for j in range(self.VOLUME_SLICES):
            X[j,:,:,0] = cv2.resize(flair_data[:,:,j+self.VOLUME_START_AT], (self.IMG_SIZE,self.IMG_SIZE))
            X[j,:,:,1] = cv2.resize(t1ce_data[:,:,j+self.VOLUME_START_AT], (self.IMG_SIZE,self.IMG_SIZE))

        return self.model.predict(X/np.max(X), verbose=1)

    
    def showPredictsById(self,flair_data,t1ce_data,start_slice = 60):
        graph_plots = {}    
        origImage = flair_data
        self.flair_data = flair_data
        self.t1ce_data = t1ce_data
        p = self.predictByPath(self.flair_data,self.t1ce_data)

        core = p[:,:,:,1]
        edema= p[:,:,:,2]
        enhancing = p[:,:,:,3]

        context = self.flair_data

        plt.switch_backend("AGG")

        #for original image
        plt.figure(figsize=(8,5))
        plt.title("flair image")
        plt.imshow(cv2.resize(context[:,:,60], (128,128)), cmap="gray")
        plt.tight_layout()
        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        img_png = buffer.getvalue()
        graph = base64.b64encode(img_png)
        graph = graph.decode('utf-8')
        buffer.close()
        graph_plots['original']=graph
        plt.close()



        #for all classes image
        plt.figure(figsize=(8,5))
        plt.title('all classes')
        plt.imshow(p[start_slice,:,:,1:4], cmap="gray")
        plt.tight_layout()
        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        img_png = buffer.getvalue()
        graph = base64.b64encode(img_png)
        graph = graph.decode('utf-8')
        buffer.close()
        graph_plots['all']=graph

        #for edema image
        plt.figure(figsize=(8,5)) 
        plt.title(f'{self.SEGMENT_CLASSES[1]} predicted')
        plt.imshow(edema[start_slice,:,:], cmap="gray")
        plt.tight_layout()
        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        img_png = buffer.getvalue()
        graph = base64.b64encode(img_png)
        graph = graph.decode('utf-8')
        buffer.close()
        graph_plots['edema']=graph

        #for core image
        plt.figure(figsize=(8,5))
        plt.title(f'{self.SEGMENT_CLASSES[2]} predicted')
        plt.imshow(core[start_slice,:,], cmap="gray")
        plt.tight_layout()
        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        img_png = buffer.getvalue()
        graph = base64.b64encode(img_png)
        graph = graph.decode('utf-8')
        buffer.close()
        graph_plots['core']=graph

        #for enhancing image
        plt.figure(figsize=(8,5))
        plt.title(f'{self.SEGMENT_CLASSES[3]} predicted')
        plt.imshow(enhancing[start_slice,:,], cmap="gray")
        plt.tight_layout()
        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        img_png = buffer.getvalue()
        graph = base64.b64encode(img_png)
        graph = graph.decode('utf-8')
        buffer.close()
        graph_plots['enhancing']=graph

        return graph_plots
    
        # axarr[1].imshow(p[start_slice,:,:,1:4], cmap="gray")
        # axarr[1].title('all classes')
        # axarr[2].imshow(edema[start_slice,:,:], cmap="gray")
        # axarr[2].title(f'{self.SEGMENT_CLASSES[1]} predicted')
        # axarr[3].imshow(core[start_slice,:,], cmap="gray")
        # axarr[3].title(f'{self.SEGMENT_CLASSES[2]} predicted')
        # axarr[4].imshow(enhancing[start_slice,:,], cmap="gray")
        # axarr[4].title(f'{self.SEGMENT_CLASSES[3]} predicted')

        # plt.show()
        
    # test_image_flair=nib.load("BraTS20_Training_016_flair (1).nii").get_fdata()
    # showPredictsById(case=test_image_flair)

    def unet_model(self,flair_data,t1ce_data):
        self.flair_data = flair_data
        self.t1ce_data = t1ce_data
        
        ############ load trained model ################
        return self.showPredictsById(self.flair_data,self.t1ce_data)
    

def handle_uploaded_file(f):  
    with open('segmentation/static/upload/'+f.name, 'wb+') as destination:  
        for chunk in f.chunks():  
            destination.write(chunk) 
    return f.name        



    


