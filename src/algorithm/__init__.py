from .flfl import get_flfl
from .fixmatch import get_fixmatch

from .algorithmbase import *
from config import cfg

def get_algorithm(specific_algorithm=None):
    if specific_algorithm is not None:
        algorithm = eval('get_{}()'.format(specific_algorithm))
        return algorithm

    # try catch block slow down the program, so remove.
    algorithm = eval('get_{}()'.format(cfg['control_ssl']['algorithm']))
    return algorithm
