import logging
import os
from pydrake.geometry import Meshcat


def StartMeshcat(open_window=False):
    logging.getLogger('drake').setLevel(logging.WARNING)
    meshcat = Meshcat()
    web_url = meshcat.web_url()

    print(f'Meshcat is now available at {web_url}')
    os.system(f'xdg-open {web_url}')
    return meshcat
