import logging
import os
from collections import OrderedDict
import torch
import pytorchpoints.utils.comm as comm
from pytorchpoints.utils.argument_parser import default_argument_parser
from pytorchpoints.checkpoint import PtCheckpointer
from pytorchpoints.config import get_cfg
from pytorchpoints.engine import DefaultTrainer, default_setup, hooks, launch, build_engine
from pytorchpoints.modeling import add_model_config

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    tmp_cfg = cfg.load_from_file_tmp(args.config_file)
    add_model_config(cfg, tmp_cfg)

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts) 
    
    if len(cfg.OUTPUT_DIR) == 0:
        cfg.OUTPUT_DIR = os.path.join(os.path.dirname(args.config_file), 'log')

    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def main(args):
    cfg = setup(args)

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = build_engine(cfg)
    trainer.resume_or_load(resume=args.resume)
    
    if args.eval_only:
        if trainer.test_data_loader is not None:
            res = trainer.test(trainer.cfg, trainer.model, trainer.test_data_loader, trainer.evaluator, epoch=-1)
        if comm.is_main_process():
            print(res)
        return res

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
