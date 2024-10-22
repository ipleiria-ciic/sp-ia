"""
File: ExportImages.py
Author: José Areia
Date: 2024-07-25
"""

# Utils
from Utils import *

# Global constants
TotalBatches = 24
GlobalCounter = 0
OutputPath = '../../Datasets/Images'

# Create or clear the mapping file at the start
with open('TRM_Dataset/AdversarialClasses_Mapping.txt', 'w') as f:
    f.write('')

for BatchIndex in range(TotalBatches):
    GlobalCounter = export_images(BatchIndex, OutputPath, GlobalCounter)