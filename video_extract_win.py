from pylab import *
from sklearn.svm import OneClassSVM
from sklearn.externals import joblib
from os.path import basename
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager
import sys, os, imtools, shutil, pickle, sift, dsift, time, getopt

typeFeats='dsift'
load_model = 'test.pkl'    # limit 24 character (include .pkl)
filename = 'test.avi'


def print_error():
  print("Usage: %s -f test_video" % sys.argv[0])
  print("    -f : File of test video")
  sys.exit(1)
  

def read_feature_labels(path):	
  featlist = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.' + typeFeats)]
  features = []

  for featfile in featlist:
    l, d = sift.read_features_from_file(featfile)
    features.append(d.flatten())

  features = array(features)
  labels = [featfile.split('/')[-1][0] for featfile in featlist]

  return features, featlist


def copyFile(src, dest):
    try:
        shutil.copy(src, dest)

    # eg. src and dest are the same file
    except shutil.Error as e:
        print('Error: %s' % e)

    # eg. source or destination doesn't exist
    except IOError as e:
        print('Error: %s' % e.strerror)


# Read command line args
opts, args = getopt.getopt(sys.argv[1:], "f:")

###############################
# o == option
# a == argument passed to the o
###############################

for o, a in opts:
    if o == '-f':
        filename = a

    else:
        print_error()


# 1. Make Directory
dirname = os.path.join('C:\Users', os.getenv('USERNAME'), 'Desktop\Extract')

if not os.path.isdir(dirname):
    os.mkdir(dirname)


# 2. Save Screen Shot to Directory
filter = 'scene'
format = 'jpg'
rate = 5
vout = 'dummy'
prefix = 'img_'
ratio = 200

cmmd = str('vlc ' + filename + ' --video-filter=scene --scene-format=jpg --rate=5 --vout=dummy --scene-prefix=extract --scene-ratio=200 --scene-path=' + dirname)
os.system(cmmd)


load_model = raw_input("Input load model name : ")

if not load_model or not os.path.isfile(load_model):
    print "No/invalid load model specified"
    print_error()
    

# 3. Extract Features
imlist = [os.path.join(dirname, f) for f in os.listdir(dirname) if f.endswith('.jpg')]		# create a list of images

for filename in imlist:
    featfile = filename[:-3] + typeFeats
    featfile = featfile.replace(" ", "")

    try:
      if not os.path.isfile(featfile):
        dsift.process_image_dsift(filename, featfile, 10, 5, resize=(100, 100))		# process images at fixed size (100, 100)

    except Exception:
      pass
    

# 4. Run Classifier against
svm = pickle.load(open(load_model, "rb")) 	# load to model file
svm_test_features, filelist = read_feature_labels(dirname)
svm_test_result = svm.predict(svm_test_features)

result_dir = os.path.join('C:\Users', os.getenv('USERNAME'), 'Desktop\Result-Video')

if not os.path.isdir(result_dir):
    os.mkdir(result_dir)

valid = 0
errors = 0

for i in range(len(svm_test_result)):
  if (svm_test_result[i] == 1):
    valid = valid + 1
    src = filelist[i][:-6] + '.jpg'

    copyFile(src, result_dir)

  else:
    errors = errors + 1


print "Valid %s" % valid
print "Errors %s" % errors
print "%d Images founded." % valid

now = time.localtime()

f = open(os.path.join(result_dir, 'Result.txt'), 'a')
result = "%04d-%02d-%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec) + '  ' + 'Video Classify Result : ' + str(valid) + '\n'
f.write(result)

f.close()

print "End"
