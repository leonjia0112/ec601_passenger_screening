#!/usr/bin/env python

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


body_zone = plt.imread('body_zones.png')
fig, ax = plt.subplots(figsize = (15, 15))
ax.imshow(body_zone)
