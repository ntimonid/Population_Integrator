import sys
import os
import re
import glob
import io
import time
import base64
import random
import tempfile
import subprocess
import nibabel
import json
import nrrd
import numpy
import cv2
import urllib.parse
import xml.etree.ElementTree as ET
import nibabel as nib
import numpy as np
import pickle as pk
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image, ImageOps
from lxml import etree
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from collections import OrderedDict
from IPython.core.debugger import set_trace
from copy import deepcopy
from itertools import cycle, islice
from PIL import Image
from IPython.core.display import display, HTML
from IPython.display import display, Javascript, clear_output, SVG
from matplotlib import cm
from IPython.display import SVG
import scipy.spatial as spatial
from skimage import morphology

Image.MAX_IMAGE_PIXELS = 933120000 # absurdly high number to avoid Python complaints due to image size
