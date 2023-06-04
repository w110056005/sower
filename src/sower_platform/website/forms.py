from django import forms

class ManagementForm(forms.Form):
    TrainingStart = forms.CharField(widget=forms.HiddenInput, initial='TrainingStart')
    UpgradeSeed = forms.CharField(widget=forms.HiddenInput, initial='UpgradeSeed')

