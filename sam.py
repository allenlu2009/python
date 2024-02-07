import comet_ml

comet_ml.init(api_key="STPC65ZP5ZZfMHYVzYQnZNWCg", project_name="if you want")

# Create the Comet Experiment for logging
exp = comet_ml.Experiment()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import math
from typing import List
from PIL import Image
from skimage import measure
import cv2
import termios
from google.colab.patches import cv2_imshow

import torch
import transformers
import accelerate

from diffusers import StableDiffusionInpaintPipeline
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator

import sys

sys.path.append("..")

HOME = os.getcwd()
print("HOME:", HOME)