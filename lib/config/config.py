from . import yacs
from .yacs import CfgNode as CN
import argparse
import os
from termcolor import colored


cfg = CN()

# experiment name
cfg.exp_name = 'p4s6'

# network
cfg.distributed = False
cfg.uv_dim = 24
cfg.use_bg = False
cfg.code_dim = 16
cfg.latent_square = 32

# loss
cfg.use_vggLoss = True
cfg.use_TL2Loss = True
cfg.vggLoss_weight = 5e-2
cfg.TLoss_weight = 1e-1

cfg.iLossMax = 1e-1
cfg.iLossMin = 1e-3
cfg.uvLossMax = 1.
cfg.uvLossMin = 5e-2
cfg.exp_k = 4e-2

# data
cfg.training_view = [0]
cfg.test_view = []
cfg.begin_ith_frame = 0
cfg.num_train_frame = 1
cfg.frame_interval = 1
cfg.nv = 6890  # number of vertices
cfg.pose_dim = 72
cfg.vertices = 'vertices'
cfg.params = 'params_4views_5e-4'
cfg.densepose = 'densepose'
cfg.mask = 'mask_cihp'
cfg.mask_bkgd = True
cfg.H = 1024
cfg.W = 1024
cfg.box_padding = 0.05
cfg.ignore_boundary = False

# metric
cfg.use_lpips = False

# texture
cfg.texture_size = 256

# task
cfg.task = 'UV_Volumes'

# model
cfg.view_noise_weight = 0.1

# gpus
cfg.gpus = list(range(8))
# if load the pretrained network
cfg.resume = True

# epoch
cfg.ep_iter = -1
cfg.save_ep = 100
cfg.save_latest_ep = 5
cfg.eval_ep = 100

# -----------------------------------------------------------------------------
# train
# -----------------------------------------------------------------------------
cfg.train = CN()

cfg.train.dataset = 'CocoTrain'
cfg.train.epoch = 10000
cfg.train.num_workers = 8
cfg.train.collator = ''
cfg.train.batch_sampler = 'default'
cfg.train.sampler_meta = CN({'min_hw': [256, 256], 'max_hw': [480, 640], 'strategy': 'range'})
cfg.train.shuffle = True
cfg.train_texture = True

# use adam as default
cfg.train.optim = 'adam'
cfg.train.lr = 5e-5
cfg.train.weight_decay = 0

cfg.train.scheduler = CN({'type': 'multi_step', 'milestones': [80, 120, 200, 240], 'gamma': 0.5})

cfg.train.batch_size = 4

# test
cfg.test = CN()
cfg.test.dataset = 'CocoVal'
cfg.test.batch_size = 1
cfg.test.epoch = -1
cfg.test.sampler = 'default'
cfg.test.batch_sampler = 'default'
cfg.test.sampler_meta = CN({'min_hw': [480, 640], 'max_hw': [480, 640], 'strategy': 'origin'})
cfg.test.frame_sampler_interval = 50


# trained model
cfg.trained_model_dir = 'data/trained_model'

# recorder
cfg.record_dir = 'data/record'
cfg.record_interval = 20

# result
cfg.result_dir = 'data/result'

# config file
cfg.cfg_dir = 'data/config'

# evaluation
cfg.skip_eval = False
cfg.use_nb_mask_at_box = False

cfg.batch_rays = 1
cfg.fix_random = False
cfg.white_bkgd = False
cfg.output_depth = False
cfg.erode_msk = False
cfg.mask_border = 0
cfg.cihp_border = 5

cfg.T_threshold = 0.5
cfg.mask_threshold = 0.1
cfg.save_img = False

cfg.evaluate = 'evaluate'
cfg.save_frame = 1

def parse_cfg(cfg, args):
    if len(cfg.task) == 0:
        raise ValueError('task must be specified')

    # assign the gpus
    os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join([str(gpu) for gpu in cfg.gpus])
    cfg.trained_model_dir = os.path.join(cfg.trained_model_dir, cfg.task, cfg.exp_name)
    cfg.record_dir = os.path.join(cfg.record_dir, cfg.task, cfg.exp_name)
    cfg.result_dir = os.path.join(cfg.result_dir, cfg.task, cfg.exp_name)
    cfg.cfg_dir = os.path.join(cfg.cfg_dir, cfg.task, cfg.exp_name)
    cfg.local_rank = args.local_rank
    cfg.device = args.device
    cfg.distributed = cfg.distributed or args.launcher not in ['none']
    if args.test:
        cfg.result_dir = os.path.join(cfg.result_dir, 'test')


def make_cfg(args):
    with open(args.cfg_file, 'r') as f:
        current_cfg = yacs.load_cfg(f)

    if 'parent_cfg' in current_cfg.keys():
        with open(current_cfg.parent_cfg, 'r') as f:
            parent_cfg = yacs.load_cfg(f)
        cfg.merge_from_other_cfg(parent_cfg)

    cfg.merge_from_other_cfg(current_cfg)
    cfg.merge_from_list(args.opts)

    parse_cfg(cfg, args)

    # remove past cfg file and save present cfg file
    if not cfg.resume and len(args.type) == 0:
        print(colored('remove cfg directory %s' % cfg.cfg_dir, 'red'))
        os.system('rm -rf {}'.format(cfg.cfg_dir))
        if not os.path.exists(cfg.cfg_dir):
            try:
                os.makedirs(cfg.cfg_dir)
                os.system('cp %s %s'% (args.cfg_file, cfg.cfg_dir))
            except Exception as e:
                pass
    return cfg


parser = argparse.ArgumentParser()
parser.add_argument("--cfg_file", default="configs/default.yaml", type=str)
parser.add_argument('--test', action='store_true', dest='test', default=False)
parser.add_argument("--type", type=str, default="")
parser.add_argument('--det', type=str, default='')
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--launcher', type=str, default='none', choices=['none', 'pytorch'])
parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
args = parser.parse_args()
if len(args.type) > 0:
    cfg.task = "run"
cfg = make_cfg(args)
