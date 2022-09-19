# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Detection Training Script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import os
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel

import warnings
warnings.simplefilter("ignore", UserWarning)

import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.utils.events import EventStorage
from detectron2.evaluation import (
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.utils.logger import setup_logger

from adet.data.dataset_mapper import DatasetMapperWithBasis
from adet.data.fcpose_dataset_mapper import FCPoseDatasetMapper
from adet.config import get_cfg
from adet.checkpoint import AdetCheckpointer
from adet.evaluation import TextEvaluator

# Some basic setup:
# Ref: https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5#scrollTo=UkNbUzUOLYf0
import numpy as np
import glob
import os
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from pycocotools import mask as mask_util

def get_mots(data_split):
    # https://www.vision.rwth-aachen.de/page/mots
    # class_id: car 1, pedestrian 2
    # format: time_frame id class_id img_height img_width rle

    # balloon sample: {"34020010494_e5cb88e1c4_k.jpg1115004":{"fileref":"","size":1115004,"filename":"34020010494_e5cb88e1c4_k.jpg","base64_img_data":"","file_attributes":{},"regions":{"0":{"shape_attributes":{"name":"polygon","all_points_x":[1020,1000,994,1003,1023,1050,1089,1134,1190,1265,1321,1361,1403,1428,1442,1445,1441,1427,1400,1361,1316,1269,1228,1198,1207,1210,1190,1177,1172,1174,1170,1153,1127,1104,1061,1032,1020],"all_points_y":[963,899,841,787,738,700,663,638,621,619,643,672,720,765,800,860,896,942,990,1035,1079,1112,1129,1134,1144,1153,1166,1166,1150,1136,1129,1122,1112,1084,1037,989,963]},"region_attributes":{}}}},"25899693952_7c8b8b9edc_k.jpg814535":{"fileref":"","size":814535,"filename":"25899693952_7c8b8b9edc_k.jpg","base64_img_data":"","file_attributes":{},"regions":{"0":{"shape_attributes":{"name":"polygon","all_points_x":[586,510,441,389,331,268,196,158,135,156,210,307,403,437,443,425,448,481,506,513,503,511,502,583,669,755,800,811,803,784,755,717,668,620,586],"all_points_y":[133,116,115,126,149,185,261,339,438,560,664,784,868,893,929,964,964,961,965,958,936,910,896,836,752,636,538,454,392,329,270,220,175,149,133]},"region_attributes":{}}}}

    # mots sample: 1 2002 2 1080 1920 UkU\1`0RQ1>PoN\OVP1X1F=I3oSOTNlg0U2lWOVNng0m1nWOWNlg0n1PXOWNlg0l1SXOUNjg0P2RXORNfg0V2WXOoMbg0V2\XOlM^g0Z2^XOkM_g0W2^XOlM]g0Y2`XOjM]g0Y2`XOkM[g0Y2aXOkM^g0V2`XOlM^g0T2aXOoM\g0m1hXOVNWg0^1eVOeLU2o1Ug0P1`YORO^f0n0cYOTO[f0j0eYOYOZf0e0hYO\OWf0=oYOEoe04XZONge00[ZO3be0K`WOnM@HY2a2ff0J]WOXNk1P2ff0J\WOYNk1P2if0GXWO]Nm1n1jf0GTWObNm1j1nf0ERWOdNn1h1of0o0kXOUOUg0m0fXOVOYg0m0cXOUO]g0o0\XOTObg0o0\XOROdg0Q1WXOROgg0P1UXOSOjg0nNXWOBh0e0[OUNeh0c1YWOEd0c0CQN_h0h1ZWOEb0b0GPN[h0k1\WOE=a00nMVh0l1[WOI<=4nMTh0Y7oWOeHQh0f1]WO[3b0oJPh0g1aWOT3c0VKjg0f1eWOQ3b0YKhg0g1fWOn2e0ZKeg0h1dWOn2i0ZKcg0h1cWOo2k0XKag0j1cWOm2n0YK_g0j1bWOn2o0XK_g0j1bWOj2U1ZKWg0n1dWOf2W1\KTg0o1eWOb2Z1_KQg0o1eWOa2[1`Kof0P2fWO]2^1cKkf0Q2gWO[2_1dKjf0P2hWOZ2`1fKff0R2jWOT2e1gKbf0V2iWOP2i1iK\f0Z2jWOe1ni0PLYVOj6ai0h0O3O1O100N4N[K`VOT1]i0lNdVOV1Zi0iNgVOY1Wi0dNkVO_1Si0aNkVOd1Qi0\NoVOh1nh0XNQWOk1mh0UNRWOm1mh0c3L1O2N3L5L1O2N2N1O4L1O4L10iHUXO`4jg0[K[XOf4dg0WK_XOk4`g0UJ]XOWO3T33A\g0RNaXOVO0S3;CUg0RNaXOo2e0mNgf0TNdXOj2n0oN^f0UNeXOb2\1VOne0WNgXOW1@nNP2d1ke0QNgXOW1JjNm1k1de0oMgXO[1J^NW2X2_e0cMdXOf1KVNZ2^2[f0\OfZOa0Ye0]OkZOa0Se0@S[Ob0dd0^O`[Oa0]d0@k[O:Rd0Ho[O5Rd0Mo[O2oc0OP\O1oc01Q\ONQd03n[OMPd03Q\OLoc06P\OMlc03T\OMmc02U\OMjc03V\OLlc05R\OKPd03Q\OKQd04o[OLWd00h[ONZd01h[OL^d0Oe[OM^d00c[OOad0O_[OOdd0N^[O0fd0MZ[O2jd0JZ[O2hd0MY[O1kd0LV[O2kd0MW[O1md0LS[O2od0MR[O2Qe0NmZO0Ue00jZONYe01iZOMZe03dZOK]e05dZOI_e07bZOF_e09bZOEbe09^ZOFbe0<\ZODfe0;YZOBme0<RZODoe0<PZOCUf0=gYOCZf0?cYOA`f0>^YOAdf0?[YOAff0a0WYO]Olf0c0SYO]OPg0d0mXO[OTg0h0hXOWOZg0i0eXOWO_g0f0`XOYObg0f0`XOXOdg0d0\XO\Oeg0c0[XO\Ogg0d0YXO[Ogg0e0ZXOXOjg0h0c4O2N2L4M7E^l`=
    folder_path = "/media/catchall/starplan/Dissertation/Dataset/MOTSChallenge/train/"
    text_file = os.path.join(folder_path, f"{data_split}.txt")
    dataset_dicts = []
    data_dict = {}
    with open(text_file) as f:
        for line in f:
            fields = line.strip().split(" ")
            try:
                group = int(fields[0])
                image_id = int(fields[1]) # 2 - image basename
                class_id = int(fields[4]) # 3 - frame id
                height = int(fields[5])
                width = int(fields[6])
                bbox_int = list(map(int, fields[7:11]))
                rle = {'size': [height, width], 'counts': fields[11].encode(encoding='UTF-8')}
                if image_id in data_dict.keys():
                    data_dict[image_id]["annotations"].append({
                        "bbox": bbox_int,
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "segmentation": rle,
                        "category_id": class_id
                    })
                else:
                    data_dict[image_id] = {}
                    data_dict[image_id]["file_name"] = f"{folder_path}new_images/{fields[0]}/{fields[2]}"
                    data_dict[image_id]["image_id"] = image_id
                    data_dict[image_id]["height"] = height
                    data_dict[image_id]["width"] = width
                    data_dict[image_id]["annotations"] = [
                        {   
                            "bbox": bbox_int,
                            "bbox_mode": BoxMode.XYXY_ABS,
                            "segmentation": rle,
                            "category_id": class_id
                        }
                    ]
            except IndexError as e:
                print(fields)
        for _, value in data_dict.items():
            dataset_dicts.append(value)
    return dataset_dicts


class Trainer(DefaultTrainer):
    """
    This is the same Trainer except that we rewrite the
    `build_train_loader`/`resume_or_load` method.
    """
    def build_hooks(self):
        """
        Replace `DetectionCheckpointer` with `AdetCheckpointer`.

        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.
        """
        ret = super().build_hooks()
        for i in range(len(ret)):
            if isinstance(ret[i], hooks.PeriodicCheckpointer):
                self.checkpointer = AdetCheckpointer(
                    self.model,
                    self.cfg.OUTPUT_DIR,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                )
                ret[i] = hooks.PeriodicCheckpointer(self.checkpointer, self.cfg.SOLVER.CHECKPOINT_PERIOD)
        return ret
    
    def resume_or_load(self, resume=True):
        checkpoint = self.checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=resume)
        if resume and self.checkpointer.has_checkpoint():
            self.start_iter = checkpoint.get("iteration", -1) + 1

    def train_loop(self, start_iter: int, max_iter: int):
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        logger = logging.getLogger("adet.trainer")
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            self.before_train()
            for self.iter in range(start_iter, max_iter):
                self.before_step()
                self.run_step()
                self.after_step()
            self.after_train()

    def train(self):
        """
        Run training.

        Returns:
            OrderedDict of results, if evaluation is enabled. Otherwise None.
        """
        self.train_loop(self.start_iter, self.max_iter)
        if hasattr(self, "_last_eval_results") and comm.is_main_process():
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable

        It calls :func:`detectron2.data.build_detection_train_loader` with a customized
        DatasetMapper, which adds categorical labels as a semantic mask.
        """
        if cfg.MODEL.FCPOSE_ON:
            mapper = FCPoseDatasetMapper(cfg, True)
        else:
            mapper = DatasetMapperWithBasis(cfg, True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                    ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        if evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        if evaluator_type == "text":
            return TextEvaluator(dataset_name, cfg, True, output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        if len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("adet.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)

    rank = comm.get_rank()
    setup_logger(cfg.OUTPUT_DIR, distributed_rank=rank, name="adet")

    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        AdetCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model) # d2 defaults.py
        if comm.is_main_process():
            verify_results(cfg, res)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop or subclassing the trainer.
    """
    for d in ["train", "val"]:
        DatasetCatalog.register("mots_" + d, lambda d=d: get_mots(d))

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
