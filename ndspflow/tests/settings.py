"""Settings for testing ndspflow."""

import os
from pathlib import Path

###################################################################################################
###################################################################################################

# Path Settings
TEST_DATA_PATH = os.path.join(Path(__file__).parent, 'data')

# Simulation settings
N_SECONDS = 10
FS = 500
EXP = -2
FREQ = 10
F_RANGE = (1, 40)
