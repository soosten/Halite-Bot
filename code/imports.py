import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from sklearn.cluster import OPTICS
from copy import deepcopy
import warnings

from time import time