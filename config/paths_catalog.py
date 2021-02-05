import os

from .settings import (
    ANET_FEATURES_PATH,
    CHARADES_FEATURES_PATH,
    CHARADES_OBJ_FEATURES_PATH,
    YOUCOOKII_FEATURES_PATH,
    YOUCOOKII_OBJ_FEATURES_PATH,
    EMBEDDINGS_PATH,
    ANNOTATIONS_PATH,)

class DatasetCatalog(object):
    DATA_DIR = "datasets"

    DATASETS = {
        "anet_cap_train": {
            "feature_path": os.path.join(ANET_FEATURES_PATH, 'ANet_240/training'),
            "ann_file_path": os.path.join(ANNOTATIONS_PATH, 'anet-cap/ANetCap_training_tokens.json'),
            "embeddings_path": os.path.join(EMBEDDINGS_PATH, 'glove.840B.300d.txt'),
        },

        "anet_cap_test": {
            "feature_path": os.path.join(ANET_FEATURES_PATH, 'ANet_240/validation'),
            "ann_file_path": os.path.join(ANNOTATIONS_PATH, 'anet-cap/ANetCap_validation_tokens.json'),
            "embeddings_path":os.path.join(EMBEDDINGS_PATH, 'glove.840B.300d.txt'),
        },
        "charades_sta_train": {
            "feature_path": os.path.join(CHARADES_FEATURES_PATH),
            "ann_file_path":os.path.join(ANNOTATIONS_PATH, 'charades-sta/charades_sta_train_tokens_w_objects.json'),
            "embeddings_path": os.path.join(EMBEDDINGS_PATH, 'glove.840B.300d.txt'),
            "obj_features": CHARADES_OBJ_FEATURES_PATH
        },

        "charades_sta_test": {
            "feature_path": os.path.join(CHARADES_FEATURES_PATH),
            "ann_file_path": os.path.join(ANNOTATIONS_PATH, 'charades-sta/charades_sta_test_tokens_w_objects.json'),
            "embeddings_path": os.path.join(EMBEDDINGS_PATH, 'glove.840B.300d.txt'),
            "obj_features": CHARADES_OBJ_FEATURES_PATH
        },
        "youcookII_train" : {
            "feature_path": YOUCOOKII_FEATURES_PATH,
            "ann_file_path" :  os.path.join(ANNOTATIONS_PATH, 'youcookII/youcookII_training_tokens_w_objects.json'),
            "embeddings_path" :  os.path.join(EMBEDDINGS_PATH, 'glove.840B.300d.txt'),
            "obj_features": YOUCOOKII_OBJ_FEATURES_PATH
            },

        "youcookII_test" : {
            "feature_path": YOUCOOKII_FEATURES_PATH,
            "ann_file_path" :  os.path.join(ANNOTATIONS_PATH, 'youcookII/youcookII_validation_tokens_w_objects.json'),
            "embeddings_path" :  os.path.join(EMBEDDINGS_PATH, 'glove.840B.300d.txt'),
            "obj_features": YOUCOOKII_OBJ_FEATURES_PATH
            },
    }

    @staticmethod
    def get(name):
        if "charades_sta" in name:
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                features_path   = attrs["feature_path"],
                obj_feat_path   = attrs["obj_features"],
                ann_file_path   = attrs["ann_file_path"],
                embeddings_path = attrs["embeddings_path"],
            )
            return dict(
                factory="CHARADES_STA",
                args=args,
            )
        if "anet_cap" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                features_path=os.path.join(data_dir, attrs["feature_path"]),
                ann_file_path=os.path.join(data_dir, attrs["ann_file_path"]),
                embeddings_path=os.path.join(data_dir, attrs["embeddings_path"]),
            )
            return dict(
                factory="ANET_CAP",
                args=args,
            )
        if "youcook" in name:
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                features_path   = attrs["feature_path"],
                obj_feat_path   = attrs["obj_features"],
                ann_file_path   = attrs["ann_file_path"],
                embeddings_path = attrs["embeddings_path"],
            )
            return dict(
                factory="YOUCOOKII",
                args=args,
            )
        raise RuntimeError("Dataset not available: {}".format(name))