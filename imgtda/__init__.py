"""
TDA Image Analysis Project: Topological tools for 2D image analysis
================================================================================
Documentation is available in the docstrings.

Contents
--------
imgtda imports from the core subpackage automatically, and in
addition provides:

Subpackages
-----------
Using any of these subpackages requires an explicit import.  For example,
``import imgtda.locallystriped``.
::
 locallystriped               --- Specialize to (smooth) locally-striped images

Utility tools
-------------
::
 test              --- Run imgtda unittests
"""

from core import Project
from core.image import Image
from core import image
from core import standard_analysis

