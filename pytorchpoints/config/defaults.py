from .config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

# The version number, to upgrade from old configs to new ones if any
# changes happen. It's recommended to keep a VERSION in your config file.
_C.VERSION = 1

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CN()

_C.DATASET.NAME = ''

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()

_C.DATALOADER.TRAIN_BATCH_SIZE = 16

_C.DATALOADER.TEST_BATCH_SIZE = 16

_C.DATALOADER.NUM_WORKERS = 4

_C.DATALOADER.DATA_ROOT = ''

_C.DATALOADER.NUM_POINTS = 10000

_C.DATALOADER.SUBSAMPLING = 0.02

_C.DATALOADER.NUM_PARTS = [15, 9, 39, 11, 7, 4, 5, 10, 12, 10, 41, 6, 7, 24, 51, 11, 6]

_C.DATALOADER.IN_RADIUS = 0.0

_C.DATALOADER.NUM_STEPS = -1

_C.DATALOADER.COLOR_DROP = 0.2

_C.DATALOADER.SHUFFLE = True

# -----------------------------------------------------------------------------
# Engine
# -----------------------------------------------------------------------------
_C.ENGINE = CN()

_C.ENGINE.NAME = 'DefaultTrainer'

_C.ENGINE.BEST_METRICS = ['ACC@1']

_C.ENGINE.FP16 = False

# -----------------------------------------------------------------------------
# Transforms
# -----------------------------------------------------------------------------
_C.AUG = CN()

_C.AUG.SCALE_LOW = 0.6

_C.AUG.SCALE_HIGH = 1.4

_C.AUG.NOISE_STD = 0.002

_C.AUG.NOISE_CLIP = 0.05

_C.AUG.TRANSLATE_RANGE = 0.0

_C.AUG.X_ANGLE_RANGE = 0.0

_C.AUG.Y_ANGLE_RANGE = 0.0

_C.AUG.Z_ANGLE_RANGE = 0.0

_C.AUG.NUM_VOTES = 10

_C.AUG.TEST_ROTATE = True

_C.AUG.BATCH_AUG = False

_C.AUG.AUGMENT_SYMMETRIES = [0, 0, 0]

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()

_C.MODEL.DEVICE = "cuda"

_C.MODEL.META_ARCHITECTURE = ''

_C.MODEL.INFEATS_DIM = 3

_C.MODEL.NUM_CLASSES = 40

_C.MODEL.WEIGHTS = ''

_C.MODEL.USE_EMA = False

_C.MODEL.EMA_DECAY = 0.9999

_C.MODEL.DROP_PATH = 0.0

_C.MODEL.INIT_GAMMA1 = 1.0

_C.MODEL.INIT_GAMMA2 = 1.0

############ STEM ############
_C.MODEL.STEM = CN()

_C.MODEL.STEM.NAME = 'ConvStem'

_C.MODEL.STEM.WIDTH = 72

_C.MODEL.STEM.FEAT_TYPE = 'concat'

_C.MODEL.STEM.USE_BNACT = True

############ BACKBONE ############
_C.MODEL.BACKBONE = CN()

_C.MODEL.BACKBONE.NAME = 'ResNet'

_C.MODEL.BACKBONE.LAYERS = [1, 2, 2, 2, 2]

_C.MODEL.BACKBONE.DIMS = [144, 288, 576, 1152, 2304]

_C.MODEL.BACKBONE.BTNK_RATIOS = [2, 2, 2, 2, 2]

_C.MODEL.BACKBONE.NPOINTS = [-1, 2048, 512, 128, 32 ]

_C.MODEL.BACKBONE.NSAMPLES = [ 20, 31, 38, 36, 34 ]

_C.MODEL.BACKBONE.RADIUS = 0.05

_C.MODEL.BACKBONE.SAMPLEDL = 0.02

############ LOCAL AGG ############
_C.MODEL.LA_TYPE = CN()

_C.MODEL.LA_TYPE.NAME = 'PointWiseMLP'

_C.MODEL.LA_TYPE.FEAT_TYPE = 'dp_fi_df'

_C.MODEL.LA_TYPE.NUM_MLPS = 1

_C.MODEL.LA_TYPE.REDUCTION = 'max'

############ DOWN_SAMPLER ############
_C.MODEL.DOWN_SAMPLER = CN()

_C.MODEL.DOWN_SAMPLER.NAME = 'MaxPoolDS'

############ HEAD ############
_C.MODEL.HEAD = CN()

_C.MODEL.HEAD.NAME = 'ClsHead'

############ GLOBAL_POOLER ############
_C.MODEL.GLOBAL_POOLER = CN()

_C.MODEL.GLOBAL_POOLER.NAME = 'GlobalAvgPooler'

############ GAMMA ############
_C.MODEL.GAMMA = CN()

_C.MODEL.GAMMA.START = -1

# -----------------------------------------------------------------------------
# RRI
# -----------------------------------------------------------------------------
_C.MODEL.RRI = CN()

_C.MODEL.RRI.NAME = ''

_C.MODEL.RRI.DIMS = [72]

_C.MODEL.RRI.NSAMPLES = [50]

_C.MODEL.RRI.NORM_LAST = True

_C.MODEL.RRI.ACT_LAST = True

_C.MODEL.RRI.REDUCTION = 'max'

# ----------------------------------------------------------------------------
# Solver
# ----------------------------------------------------------------------------
_C.SOLVER = CN()

_C.SOLVER.NAME = 'Adam'

_C.SOLVER.EPOCH = 10

_C.SOLVER.CHECKPOINT_PERIOD = 1

_C.SOLVER.EVAL_PERIOD = 1

_C.SOLVER.VOTE_START = 1

_C.SOLVER.DENSE_EVAL_EPOCH = 1000000

_C.SOLVER.AUTO_LR = True

_C.SOLVER.BASE_LR = 0.0005

_C.SOLVER.BIAS_LR_FACTOR = 1.0

_C.SOLVER.WEIGHT_DECAY = 0.0

_C.SOLVER.WEIGHT_DECAY_NORM = 0.0

_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0

_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.ALPHA = 0.99

_C.SOLVER.BETAS = [0.9, 0.999]

_C.SOLVER.EPS = 1e-8

_C.SOLVER.GRAD_CLIP_TYPE = 'norm' # norm, value

_C.SOLVER.GRAD_CLIP = -1.0 # ignore <= 0

_C.SOLVER.NORM_TYPE = 2.0

_C.SOLVER.WRITE_PERIOD = 20

# ----------------------------------------------------------------------------
# lr scheduler
# ----------------------------------------------------------------------------
_C.LR_SCHEDULER = CN()

_C.LR_SCHEDULER.NAME = 'StepLR'

_C.LR_SCHEDULER.STEP_SIZE = 3

_C.LR_SCHEDULER.GAMMA = 0.1

_C.LR_SCHEDULER.MILESTONES = (3,)

_C.LR_SCHEDULER.WARMUP_EPOCH = -1

# ---------------------------------------------------------------------------- #
# Losses
# ---------------------------------------------------------------------------- #
_C.LOSSES = CN()

_C.LOSSES.NAMES = ['LabelSmoothing']

_C.LOSSES.LABELSMOOTHING = 0.2

# ---------------------------------------------------------------------------- #
# Inference
# ---------------------------------------------------------------------------- #
_C.INFERENCE = CN()

_C.INFERENCE.NAME = ''

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = ""

_C.SEED = -1

_C.CUDNN_BENCHMARK = True
