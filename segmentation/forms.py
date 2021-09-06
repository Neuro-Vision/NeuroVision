from django import forms

class UploadFile(forms.Form):  
    flair_field = forms.FileField(label = "Flair Image : ")
    t1ce_field = forms.FileField(label = "T1 Ce Image : ")