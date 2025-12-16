###########################################################################
# Created by: NTU
# Email: heshuting555@gmail.com
# Copyright (c) 2023
###########################################################################

import os
from .mevis import register_mevis_instances

# ====    Predefined splits for EndoVis-IM17    ===========
_PREDEFINED_SPLITS_EV17 = {
    "EndoVisIM17_train": ("EndoVis-IM17/train",
                   "EndoVis-IM17/train/meta_expressions.json"),
    "EndoVisIM17_val": ("EndoVis-IM17/valid_u",
                 "EndoVis-IM17/valid_u/meta_expressions.json"),
    "EndoVisIM17_test": ("EndoVis-IM17/valid",
                  "EndoVis-IM17/valid/meta_expressions.json"),
}

# ====    Predefined splits for EndoVis-IM18    ===========
_PREDEFINED_SPLITS_EV18 = {
    "EndoVisIM18_train": ("EndoVis-IM18/train",
                   "EndoVis-IM18/train/meta_expressions.json"),
    "EndoVisIM18_val": ("EndoVis-IM18/valid_u",
                 "EndoVis-IM18/valid_u/meta_expressions.json"),
    "EndoVisIM18_test": ("EndoVis-IM18/valid",
                  "EndoVis-IM18/valid/meta_expressions.json"),
}

def register_all_mevis(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_EV17.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_mevis_instances(
            key,
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )

    for key, (image_root, json_file) in _PREDEFINED_SPLITS_EV18.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_mevis_instances(
            key,
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )



if __name__.endswith(".builtin"):
    # Assume pre-defined datasets live in `./datasets`.
    # _root = os.getenv("DETECTRON2_DATASETS", "datasets")
    _root = "/nfs/home/mwei/MeViS/datasets"
    register_all_mevis(_root)


