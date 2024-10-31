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
OutputPath = '../Datasets/TRM-Adversarial'

# Create or clear the mapping file at the start
with open('Dataset/AdversarialClasses_Mapping-2.txt', 'w') as f:
    f.write('')

for BatchIndex in range(TotalBatches):
    GlobalCounter = export_images(BatchIndex, OutputPath, GlobalCounter)