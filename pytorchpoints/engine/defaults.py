import time
import logging
import tqdm
import os
import itertools
import numpy as np
import weakref
from collections import OrderedDict
from typing import Dict, List, Optional
from omegaconf import OmegaConf

import torch
from torch.nn.parallel import DistributedDataParallel

from pytorchpoints.config import kfg
from pytorchpoints.utils import comm
from pytorchpoints.utils.collect_env import collect_env_info
from pytorchpoints.utils.env import TORCH_VERSION, seed_all_rng
from pytorchpoints.utils.events import CommonMetricPrinter, JSONWriter, TensorboardXWriter, get_event_storage
from pytorchpoints.utils.file_io import PathManager
from pytorchpoints.utils.logger import setup_logger
from pytorchpoints.utils.flops_counter import get_model_complexity_info

from pytorchpoints.checkpoint import PtCheckpointer
from pytorchpoints.datasets import build_points_train_loader, build_points_test_loader
from pytorchpoints.modeling import build_model
from pytorchpoints.optim import build_optimizer
from pytorchpoints.lr_scheduler import build_lr_scheduler
from pytorchpoints.losses import build_losses
from pytorchpoints.datasets.transforms import PointcloudScaleAndJitter, PointcloudTestRotate
from pytorchpoints.evaluation import build_evaluation
from pytorchpoints.functional import get_cls_features
from pytorchpoints.utils.cuda import NativeScaler

from . import hooks
from .train_loop import TrainerBase
from .build import ENGINE_REGISTRY

__all__ = [
    "default_setup",
    "default_writers",
    "DefaultTrainer",
]

def default_setup(cfg, args):
    output_dir = cfg.OUTPUT_DIR
    if comm.is_main_process() and output_dir:
        PathManager.mkdirs(output_dir)

    rank = comm.get_rank()
    setup_logger(output_dir, distributed_rank=rank, name="fvcore")
    logger = setup_logger(output_dir, distributed_rank=rank)

    logger.info("Rank of current process: {}. World size: {}".format(rank, comm.get_world_size()))
    logger.info("Environment info:\n" + collect_env_info())

    logger.info("Command line arguments: " + str(args))
    if hasattr(args, "config_file") and args.config_file != "":
        logger.info(
            "Contents of args.config_file={}:\n{}".format(
                args.config_file, PathManager.open(args.config_file, "r").read()
            )
        )

    logger.info("Running with full config:\n{}".format(cfg))
    if comm.is_main_process() and output_dir:
        # Note: some of our scripts may expect the existence of
        # config.yaml in output directory
        path = os.path.join(output_dir, "config.yaml")
        with PathManager.open(path, "w") as f:
            f.write(cfg.dump())
        logger.info("Full config saved to {}".format(path))

    # make sure each worker has a different, yet deterministic seed if specified
    seed_all_rng(None if cfg.SEED < 0 else cfg.SEED + rank)

    # cudnn benchmark has large overhead. It shouldn't be used considering the small size of
    # typical validation set.
    #if not (hasattr(args, "eval_only") and args.eval_only):
    #    torch.backends.cudnn.benchmark = cfg.CUDNN_BENCHMARK
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = cfg.CUDNN_BENCHMARK

def default_writers(output_dir: str, max_iter: Optional[int] = None):
    return [
        CommonMetricPrinter(max_iter),
        JSONWriter(os.path.join(output_dir, "metrics.json")),
        TensorboardXWriter(output_dir),
    ]

def create_ddp_model(model, *, fp16_compression=False, **kwargs):
    if comm.get_world_size() == 1:
        return model
    if "device_ids" not in kwargs:
        kwargs["device_ids"] = [comm.get_local_rank()]
    ddp = DistributedDataParallel(model, **kwargs)
    #if fp16_compression:
    #    from torch.distributed.algorithms.ddp_comm_hooks import default as comm_hooks
    #    ddp.register_comm_hook(state=None, hook=comm_hooks.fp16_compress_hook)
    return ddp    

@ENGINE_REGISTRY.register()
class DefaultTrainer(TrainerBase):
    def __init__(self, cfg):
        super().__init__()
        logger = logging.getLogger("pytorchpoints")
        if not logger.isEnabledFor(logging.INFO):
            setup_logger()

        model = self.build_model(cfg)
        self.optimizer = self.build_optimizer(cfg, model)
        self.train_data_loader = self.build_train_loader(cfg)
        self.test_data_loader = self.build_test_loader(cfg)
        self.iters_per_epoch = len(self.train_data_loader)
        self._train_data_loader_iter = iter(self.train_data_loader)
        
        self.losses = self.build_losses(cfg)
        self.scheduler = self.build_lr_scheduler(cfg, self.optimizer, self.iters_per_epoch)
        self.evaluator = build_evaluation(cfg) if self.test_data_loader is not None else None

        #with torch.no_grad():
        #    flops_count, params_count = get_model_complexity_info(model, cfg.DATALOADER.NUM_POINTS, as_strings=True,
        #        print_per_layer_stat=False, verbose=False)
        #logger.info('Model created, flops_count: %s, param count: %s' % (flops_count, params_count))

        self.model = create_ddp_model(model, broadcast_buffers=False)
        self.model.train()
        
        self.checkpointer = PtCheckpointer(
            self.model,
            cfg.OUTPUT_DIR,
            trainer=weakref.proxy(self),
        )

        if cfg.ENGINE.FP16:
            self.loss_scaler = NativeScaler()
            self.loss_scaler._scaler = torch.cuda.amp.GradScaler()
        else:
            self.loss_scaler = None

        self.best_score = 0
        self.cfg = cfg
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.EPOCH * self.iters_per_epoch
        self.register_hooks(self.build_hooks())

    def resume_or_load(self, resume=True):
        self.checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=resume)
        if resume and self.checkpointer.has_checkpoint():
            self.start_iter = self.iter + 1

    def build_hooks(self):
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(),
        ]
        if cfg.MODEL.GAMMA.START >= 0:
            ret.append(hooks.GammaScheduler(start=cfg.MODEL.GAMMA.START))

        def test_and_save_results(epoch):
            self._last_eval_results = self.test(self.cfg, self.model, self.test_data_loader, self.evaluator, epoch)
            score = sum([self._last_eval_results[k] for k in self._last_eval_results if k in self.cfg.ENGINE.BEST_METRICS])
            if self.best_score < score:
                self.best_score = score
            return self._last_eval_results

        if self.test_data_loader is not None:
            ret.append(hooks.EvalHook(cfg.SOLVER.EVAL_PERIOD, cfg.SOLVER.DENSE_EVAL_EPOCH, test_and_save_results, self.iters_per_epoch))

        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD * self.iters_per_epoch))

        if comm.is_main_process():
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=cfg.SOLVER.WRITE_PERIOD))
        return ret

    def build_writers(self):
        return default_writers(self.cfg.OUTPUT_DIR, self.max_iter)

    def train(self):
        super().train(self.start_iter, self.max_iter)

    @classmethod
    def build_model(cls, cfg):
        model = build_model(cfg)
        logger = logging.getLogger(__name__)
        logger.info("Model:\n{}".format(model))
        return model

    @classmethod
    def build_optimizer(cls, cfg, model):
        return build_optimizer(cfg, model)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer, iters_per_epoch):
        return build_lr_scheduler(cfg, optimizer, iters_per_epoch)

    @classmethod
    def build_train_loader(cls, cfg):
        return build_points_train_loader(cfg)

    @classmethod
    def build_test_loader(cls, cfg):
        return build_points_test_loader(cfg)

    @classmethod
    def build_losses(cls, cfg):
        return build_losses(cfg)

    def state_dict(self):
        ret = super().state_dict()
        ret["optimizer"] = self.optimizer.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.optimizer.load_state_dict(state_dict["optimizer"])

    def _write_metrics(
        self,
        loss_dict: Dict[str, torch.Tensor],
        data_time: float,
        prefix: str = "",
    ):
        """
        Args:
            loss_dict (dict): dict of scalar losses
            data_time (float): time taken by the dataloader iteration
        """
        metrics_dict = {}
        for k, v in loss_dict.items():
            if isinstance(v, torch.Tensor):
                metrics_dict.update({ k: v.detach().cpu().item() })
            else:
                metrics_dict.update({ k: v })
        #metrics_dict = {k: v.detach().cpu().item() for k, v in loss_dict.items()}
        metrics_dict["data_time"] = data_time

        # Gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            storage = get_event_storage()

            # data_time among workers can have high variance. The actual latency
            # caused by data_time is the maximum among workers.
            data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
            storage.put_scalar("data_time", data_time)

            # average the rest metrics
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict]) for k in all_metrics_dict[0].keys()
            }
            total_losses_reduced = sum(metrics_dict.values())
            if not np.isfinite(total_losses_reduced):
                raise FloatingPointError(
                    f"Loss became infinite or NaN at iteration={self.iter}!\n"
                    f"loss_dict = {metrics_dict}"
                )

            storage.put_scalar("{}total_loss".format(prefix), total_losses_reduced)
            if len(metrics_dict) > 1:
                storage.put_scalars(**metrics_dict)

    @classmethod
    def test(cls, cfg, model, test_data_loader, evaluator, epoch):
        model.eval()
        with torch.no_grad():
            eval_res = evaluator.evaluate(cfg, model, test_data_loader, epoch)
        model.train()
        return eval_res

    def run_step(self):
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        
        try:
            data = next(self._train_data_loader_iter)
        except StopIteration:
            if comm.get_world_size() > 1:
                self.train_data_loader.sampler.set_epoch(self.iter//self.iters_per_epoch)     
            self._train_data_loader_iter = iter(self.train_data_loader)
            data = next(self._train_data_loader_iter)

            if not self.cfg.DATALOADER.SHUFFLE:
                self.train_data_loader.dataset._map_func._obj.epoch = self.iter//self.iters_per_epoch

        data_time = time.perf_counter() - start
        data = comm.unwrap_model(self.model).preprocess_batch(data)
        
        with torch.cuda.amp.autocast(enabled=self.cfg.ENGINE.FP16):
            logits = self.model(data)
            data.update({ kfg.LOGITS: logits })
            losses_dict = {}
            for loss in self.losses:
                #loss_dict = loss(logits, data[kfg.LABELS])
                loss_dict = loss(data)
                losses_dict.update(loss_dict)

        losses = [losses_dict[k] for k in losses_dict if 'acc' not in k]
        losses = sum(losses)
        self._write_metrics(losses_dict, data_time)

        self.optimizer.zero_grad()
        
        if self.loss_scaler is not None:
            self.loss_scaler(losses, self.optimizer)
        else:
            losses.backward()
            self.optimizer.step()