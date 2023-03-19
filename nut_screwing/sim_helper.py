import logging
from pydrake.geometry import Meshcat


def StartMeshcat(open_window=False):
    logging.getLogger('drake').setLevel(logging.WARNING)
    meshcat = Meshcat()

    return meshcat
