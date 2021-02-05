import os

CODE_ROOT = os.path.dirname(os.path.realpath(__file__))
ANNOTATIONS_PATH = os.path.join(os.path.dirname(CODE_ROOT), "preprocessing")

HOME = os.environ["HOME"]
DATA_PATH = os.path.join(HOME, "data", "DORi")

ANET_FEATURES_PATH = os.path.join(DATA_PATH, "anet_cap")
ANET_OBJ_FEATURES_PATH = os.path.join(DATA_PATH, "anet_cap_obj_features")

CHARADES_FEATURES_PATH = os.path.join(DATA_PATH, "charades_features_full", "rgb")
CHARADES_OBJ_FEATURES_PATH = os.path.join(DATA_PATH, "charades_sta_selected_features")

YOUCOOKII_FEATURES_PATH = os.path.join(DATA_PATH, "youcookII_i3d")
YOUCOOKII_OBJ_FEATURES_PATH = os.path.join(DATA_PATH, "youcookII_obj_features")

EMBEDDINGS_PATH = os.path.join(DATA_PATH, "word_embeddings")
