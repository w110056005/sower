from django import forms
# Form with a static dropdown
class VersionDropdownForm(forms.Form):
    CATEGORY_CHOICES = [
        ('latest', 'latest'),
        ('1.0.0', '1.0.0'),
        ('1.0.1', '1.0.1'),
    ]
    version = forms.ChoiceField(choices=CATEGORY_CHOICES)
