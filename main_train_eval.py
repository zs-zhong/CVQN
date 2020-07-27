import argparse
import pprint
import numpy as np
np.set_printoptions(suppress=True)
from config import config
from config import update_config, create_logger
from trainer import *

def parse_args():
    parser = argparse.ArgumentParser(description='Train Deep Compression System: CVQN')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args

def main():
    args = parse_args()
    logger, model_dir, tb_log_dir = create_logger(config, args.cfg, 'train')
    logger.info(pprint.pformat(args))
    logger.info(config)

    if config['IMP_TYPE'] == 'predefine':
        trainer = PredefineTrainer(config, logger, model_dir, tb_log_dir)

    elif config['IMP_TYPE'] == 'RE':
        trainer = RETrainer(config, logger, model_dir, tb_log_dir)

    elif config['IMP_TYPE'] == 'SE':
        trainer = SETrainer(config, logger, model_dir, tb_log_dir)

    else:
        raise NotImplementedError("Trainer type error.")
        
    for epoch in range(config['TRAIN']['NUM_EPOCH']):
        trainer.train()
        trainer.eval()

        if config['IMP_TYPE'] == 'RE' and (epoch + 1) % 10 == 0:
            trainer.re_based_get_imp()

        if epoch == config['TRAIN']['NUM_EPOCH'] - 1:
            trainer.save_checkpoint('final.pth')

        trainer.update_lr()

    trainer.writer.close()


if __name__ == '__main__':
    main()