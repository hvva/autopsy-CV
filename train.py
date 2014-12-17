from sklearn.svm import OneClassSVM
from sklearn.externals import joblib
from os.path import basename
from pylab import *
from PIL import Image
import sys, os, shutil, pickle, getopt, sift, dsift
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager

dirTrain = ''
save_model = ''    # limit 24 character (include .pkl)


def read_feature_labels(path):
  featlist = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.dsift')]
  features = []
  for featfile in featlist:
    l, d = sift.read_features_from_file(featfile)
    features.append(d.flatten())

  features = array(features)

  return features


def get_fileNameList(path):
  featlist = [os.path.join(path, f) for f in os.listdir(path)]

  return array(featlist)


def copyFile(src, dest):
    try:
        shutil.copy(src, dest)

    # eg. src and dest are the same file
    except shutil.Error as e:
        print('Error: %s' % e)

    # eg. source or destination doesn't exist
    except IOError as e:
        print('Error: %s' % e.strerror)


# png, bmp, gif changed to jpg (Train)
filelist = get_fileNameList(dirTrain)
for infile in filelist:
  temp1 = os.path.splitext(infile)[0]
  temp2 = os.path.splitext(infile)[1]
  temp1 = temp1.replace(" ", "")
  os.rename(infile, temp1 + temp2)

filelist = get_fileNameList(dirTrain)
for infile in filelist: 
    outfile = os.path.splitext(infile)[0] + ".jpg"

    if infile != outfile:
        try:
            Image.open(infile).save(outfile)

        except IOError:
            if os.path.isfile(outfile):
              os.remove(outfile)

        except Exception:
            pass


# create dsift file (Train)
imlist = [os.path.join(dirTrain, f) for f in os.listdir(dirTrain) if f.endswith('.jpg')]

for filename in imlist:
    featfile = filename[:-4] + '.dsift'
    featfile = featfile.replace(" ", "")

    try:
      if not os.path.isfile(featfile):
        dsift.process_image_dsift(filename, featfile, 10, 5, resize=(100,100)) # process images at fixed size (100,100)

    except Exception:
      pass

svm_train = read_feature_labels(dirTrain)
svm = OneClassSVM(nu=0.1, kernel="linear", gamma=0.1, coef0=0.0, shrinking=True, tol=0.001,
                  cache_size=200, verbose=False, max_iter=-1, random_state=None)
svm.fit(svm_train)
pickle.dump(svm, open(save_model, "wb"))
