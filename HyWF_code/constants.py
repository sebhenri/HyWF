from os.path import join, abspath, dirname, pardir

# Logging format
#LOG_FORMAT = "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
LOG_FORMAT = "%(message)s"

BASE_DIR = abspath(dirname(__file__))
CONFIG_FILE = BASE_DIR+"/options/config.ini"

# Characters
CSV_SEP = ';'
TRACE_SEP = '\t'
NL = '\n'  # new line

# Directions
IN = -1
OUT = 1
DIR_NAMES = {IN: "incoming", OUT: "outgoing"}
DIRECTIONS = [OUT, IN]

# AP states
GAP = 0x00
BURST = 0x01
WAIT = 0x02
DICT_STATES = {GAP: "gap", BURST: "burst", WAIT: "wait"}

# Mappings
DIRS2EP = {OUT: 'client', IN: 'server'}
EP2DIRS = {'client': OUT, 'server': IN}
MODE2STATE = {'gap': GAP, 'burst': BURST}

# Histograms
INF = float("inf")
NO_SEND_HISTO = -1

# logging levels
NONE = 7
INFO = 6
DEBUG = 5
VDEBUG = 4
ALL = 3
DICT_LOGS = {"NONE": NONE, "INFO": INFO, "DEBUG": DEBUG, "VDEBUG": VDEBUG, "ALL": ALL}
DICT_LOGS_RVS = {NONE: "NONE", INFO: "INFO", DEBUG: "DEBUG", VDEBUG: "VDEBUG", ALL: "ALL"}