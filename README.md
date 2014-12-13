autopsy-CV
==========

Autopsy Computer Vision Module developed by the BoB Outc4se team

Son Jinhyuk, U Sungkyung, Jang Hyejun and Choi Jinyoung

For more information about our research and this project, please see http://bob-safekids.blogspot.com/

## About Autopsy CV
This is a python module for Autopsy 3.1.1+ [http://www.sleuthkit.org/autopsy/]. With this, an investigator can train a *single-class* classifier, and run it to automatically sort selected images.

## Classifier
Using a single-class SVM classifer. The classifer must be trained before using the classifier within Autopsy. For example, known bad images of a similar class can be extracted into a folder, and the model can be trained from this folder to automatically detect images of the same class from an unknown set on new cases.
