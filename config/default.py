from pathlib import Path
from yacs.config import CfgNode as CN
import os
import time
import logging

_C = CN()
_C.NAME = ''
_C.GPU_DEVICE = 5
_C.PRINT_FREQ = 100
_C.WORKERS = 4
_C.LOG_DIR = 'logs'
_C.MODEL_DIR = 'ckps'
_C.TSBD_DIR = 'tbs'
_C.IN_CHNS = 3
_C.CODE_CHNS = 32
_C.QUA_LEVELS = [3, 5, 7]
_C.IMP_TYPE = 'predefine'
_C.LR_SE =  0.001
_C.RE_END_BATCH =  100
_C.DATASET = CN(new_allowed=True)
_C.TRAIN = CN(new_allowed=True)
_C.ENC = CN(new_allowed=True)
_C.DEC = CN(new_allowed=True)
_C.QUA = CN(new_allowed=True)
_C.ENP = CN(new_allowed=True)

def update_config(cfg, args):
    cfg.defrost()
    
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    cfg.freeze()

def create_logger(cfg, cfg_name, phase='train'):
    root_log_dir = Path(cfg.LOG_DIR)
    # set up logger
    if not root_log_dir.exists():
        print('=> creating {}'.format(root_log_dir))
        root_log_dir.mkdir()

    cfg_name = os.path.basename(cfg_name).split('.')[0]

    final_log_dir = root_log_dir  / cfg_name

    print('=> creating {}'.format(final_log_dir))
    final_log_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y%m%d%H%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_log_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    model_dir = Path(cfg.MODEL_DIR)  / cfg_name / (cfg_name + '_' + time_str)
    print('=> creating {}'.format(model_dir))
    model_dir.mkdir(parents=True, exist_ok=True)

    tensorboard_dir = Path(cfg.TSBD_DIR)  / cfg_name / (cfg_name + '_' + time_str)
    print('=> creating {}'.format(tensorboard_dir))
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(model_dir), str(tensorboard_dir)