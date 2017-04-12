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
from scipy import misc
import defect_analysis as da
import numpy as np

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
    self.all_defects = []
    self.defect_regions = {}
    self.classified_region = []
    self.unclassified_defect_region = []
    self.defect_free_region = []

  def classify_defects(self, sigma=3, radius=10, mode='parallel'):

    # Get the defects
    self.all_defects = da.classify_all_defects(self.bmp, self.orientation_fields[sigma].topological_defects, 
      self.persistence_h1_gens, radius, mode)

    # Combine defects by type
    self.defect_regions = da.combine_classified_defects_into_regions(self.bmp, self.all_defects)

    # Aggregate the entire classified region
    classified_region = np.zeros(self.bmp.shape)
    for region in self.defect_regions:
        classified_region = classified_region + self.defect_regions[region]
    self.classified_region = (classified_region > 0).astype(np.int)

    # Compute the unclassified region and defect-free region
    self.unclassified_defect_region = da.get_unclassified_defect_region(self.classified_region, self.orientation_fields[sigma].topological_defects, radius)
    self.defect_free_region = da.get_defect_free_region(self.classified_region, self.unclassified_defect_region)


def load_image_from_file(bmp_path):
  """
  Loads a bitmap image from a file and returns an Image object.
  """
  bmp = misc.imread(bmp_path)
  return Image(bmp)

