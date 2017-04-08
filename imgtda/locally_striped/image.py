# TDA Image Analysis Project : locally_striped.image
#
# Copyright (C) 2016-2017 TDA Image Analysis Project
# Author: Rachel Levanger <rachel.levanger@gmail.com>


"""
The TDA Image Analysis Project is an open source Python library for performing
2D image analysis via topological methods. It is meant to be used
in conjunction with
  - tda-persistence-explorer (https://github.com/rachellevanger/tda-persistence-explorer)
which can be used to generate and explore persistence diagrams of 2D images.
"""

from ..core import image

######################################################################
## Table of Contents
######################################################################
## 
## - Data Classes
##   - Image
##   
######################################################################


class Image(image.Image):
  """
  The main data container object for a single Locally-Striped image. Initialize
  by passing through a bitmap image stored as a numpy array.
  """

  def __init__(self, bmp):
    image.Image.__init__(self, bmp)
    self.test = "hi!"

