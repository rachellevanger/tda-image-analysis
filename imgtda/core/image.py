# TDA Image Analysis Project : core.image
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

import numpy as np
import standard_analysis as sa
import pandas as pd
from scipy import misc

######################################################################
## Table of Contents
######################################################################
## 
## - Data Classes
##   - Image
##   - OrientationField
##
######################################################################

class Image:
  """
  The main data container object for a single image. Initialize
  by passing through a bitmap image stored as a numpy array.

  Persistence diagrams are stored in a dictionary. Downstream functions
  expect that they will be Pandas DataFrames with the output columns given
  by the tda-persistence-explorer app.
  """

  def __init__(self, bmp):
      self.bmp = bmp
      self.orientation_fields = {}
      self.persistence_diagrams = {}

  def generate_orientation_field(self, sigma=3, generate_defects=1):
    """
    Generates the orientation field for the input image with smoothing
    parameter sigma. If generate_defects==1, then also generate the
    matrix of topological defect charges.
    """
    self.orientation_fields[sigma] = OrientationField(self.bmp, sigma)
    if generate_defects:
      self.orientation_fields[sigma].generate_topological_defects()

  def load_sublevel_pd(self, path):
    self.persistence_diagrams['sub'] = pd.DataFrame(pd.read_csv(path))

  def load_superlevel_pd(self, path):
    self.persistence_diagrams['sup'] = pd.DataFrame(pd.read_csv(path))


class OrientationField:
  """
  Container object for an orientation field and its defects. Both are stored
  as arrays of floats. The orientation field gives the orientation in
  [-pi/2, pi/2), and the topological defects are typically in {-1., 0., 1.}.
  """

  def __init__(self, bmp, sigma):
    self.orientation_field = sa.orientation_field(bmp, sigma)
    self.topological_defects = []

  def generate_topological_defects(self):
    """
    Outputs a matrix of the charges of topological defects about each point
    in the orientation field. Pad with zeros on right and bottom by unit width.
    """
    td = sa.topological_defect_array(self.orientation_field)
    td = np.vstack((td, np.zeros((td.shape[1],))))
    td = np.hstack((td, np.zeros((td.shape[0],1))))
    self.topological_defects = td


def load_image_from_file(bmp_path):
  """
  Loads a bitmap image from a file and returns an Image object.
  """
  bmp = misc.imread(bmp_path)
  return Image(bmp)

