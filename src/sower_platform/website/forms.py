from django import forms

class ManagementForm(forms.Form):
    TrainingStart = forms.CharField(widget=forms.HiddenInput, initial='TrainingStart')
    UpgradeSeed = forms.CharField(widget=forms.HiddenInput, initial='UpgradeSeed')


# Form with a static dropdown
class VersionDropdownForm(forms.Form):
    CATEGORY_CHOICES = [
        ('latest', 'latest'),
        ('1.0.0', '1.0.0'),
        ('1.0.1', '1.0.1'),
    ]
    category = forms.ChoiceField(choices=CATEGORY_CHOICES)
