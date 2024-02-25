import _init_path
import argparse
import datetime
import glob
import os
from pathlib import Path
from test import repeat_eval_ckpt

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, model_fn_decorator
from pcdet.utils import common_utils
from train_utils.optimization import build_optimizer, build_scheduler
# from train_utils.train_utils import train_model
from train_utils.train_multi_db_dg_AE_TTT import train_multi_db_model as train_model
import pickle


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=1, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=8, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    parser.add_argument('--ckpt_save_interval', type=int, default=1, help='number of training epochs')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--max_ckpt_save_num', type=int, default=30, help='max number of saved checkpoint')
    parser.add_argument('--iter_num_per_tar', type=int, default=5, required=True, help='iteration number per target batch') # 
    parser.add_argument('--learning_rate', type=float, default=0.01, required=True, help='learning_rate') # 
    parser.add_argument('--TTT_mode', type=int, default=0, required=True, help='mode of testing-time training with 0 for standard and 1 for online') # 
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, help='')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=0, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--num_epochs_to_eval', type=int, default=0, help='number of checkpoints to be evaluated')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')

    parser.add_argument('--pretrained_pointnet2', type=str, default=None, help='pretrained_pointnet2')
    parser.add_argument('--output_custom_dir', type=str, default=None, help='custom output dir')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)
    
    assert args.TTT_mode in [0, 1]
    
    cfg.OPTIMIZATION.LR = args.learning_rate

    return args, cfg


def main():
    args, cfg = parse_config()
    if args.launcher == 'none':
        print ("None args.launcher********",args.launcher)
        dist_train = False
        total_gpus = 1
    else:
        print ("args.launcher********",args.launcher)
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_train = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs

    if args.fix_random_seed:
        common_utils.set_random_seed(666)

    if args.output_custom_dir is None:
        output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    else:
        output_dir = Path(args.output_custom_dir) / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    # output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    ckpt_dir = output_dir / 'ckpt'
    eval_output_dir = output_dir / 'eval' / 'eval_TT'
    ps_label_dir = output_dir / 'ps_label'
    ps_label_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)


    log_file = output_dir / ('log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_train:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)
    if cfg.LOCAL_RANK == 0:
        os.system('cp %s %s' % (args.cfg_file, output_dir))

    tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard')) if cfg.LOCAL_RANK == 0 else None

    # -----------------------create dataloader & network & optimizer---------------------------
    source_set, source_loader, source_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.DATA_CONFIG.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers,
        logger=logger,
        training=False,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        total_epochs=args.epochs,
        shuffle_sample_training = False, # disable shuffling when training
    )


    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.DATA_CONFIG.CLASS_NAMES), dataset=source_set)

    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model.cuda()

    # load checkpoint if it is possible
    start_epoch = it = 0
    last_epoch = -1

    # laoding model_state
    logger.info('**********************Loading Pretrained Model**********************')
    model.load_params_from_file(filename=args.pretrained_model, to_cpu=dist_train, logger=logger)

    

    # freezed all modules
    frozen_layer = [ # 'vfe', 'backbone_3d', 'decoder_3d',
        'map_to_bev_module', 
        'backbone_2d', 'dense_head', 'roi_head']
    for name, param in model.named_parameters():
        for module_name in frozen_layer:
            if module_name in name:
                param.requires_grad_(False)
                break
            else:
                param.requires_grad_(True)

    # load pretrained parameters for perceptual loss calculation
    if cfg.MODEL.DECODER_3D.get('PERCEPTUAL_LOSS_CALCULATION', False):
        pointnet2_model_dict = torch.load(args.pretrained_pointnet2)['model_state_dict']
        names_pretrain = list(pointnet2_model_dict.keys())
        names_to_update = list(model.decoder_3d.perceptual_module.state_dict().keys())
        for name in names_pretrain:
            if name not in names_to_update:
                pointnet2_model_dict.pop(name)
        model.decoder_3d.perceptual_module.load_state_dict(pointnet2_model_dict)
        # froze pretrained pointnet2
        for name, param in model.decoder_3d.perceptual_module.named_parameters():
            param.requires_grad_(False)

    if cfg.OPTIMIZATION.get('TEST_TIME_BN', False):
        for name, module in model.named_modules():
            if ('vfe' in name) or ('backbone_3d' in name) or ('decoder_3d' in name) or ('backbone_2d' in name):
                if isinstance(module, torch.nn.BatchNorm2d) or isinstance(module, torch.nn.BatchNorm1d):
                    # print('modules:', type(module), name, '\tmean:', module.running_mean, '\tvar:', module.running_var )
                    module.running_mean = None
                    module.running_var = None
                    # print('after modification, modules:', name, '\tmean:', module.running_mean, '\tvar:', module.running_var )
                    # module.track_running_stats = False
    
    

    optimizer = build_optimizer(model, cfg.OPTIMIZATION)

    # logger.info('**********************Loading Discriminator Model**********************')
    # model.load_params_from_file(filename=args.pretrained_discriminator, to_cpu=dist_train, logger=logger)

    logger.info('**********************Finished Loading Models**********************')

    # if args.ckpt is not None:
    #     it, start_epoch = model.load_params_with_optimizer(args.ckpt, to_cpu=dist_train, optimizer=optimizer, logger=logger)
    #     last_epoch = start_epoch + 1
    # else:
    #     ckpt_list = glob.glob(str(ckpt_dir / '*checkpoint_epoch_*.pth'))
    #     if len(ckpt_list) > 0:
    #         ckpt_list.sort(key=os.path.getmtime)
    #         it, start_epoch = model.load_params_with_optimizer(
    #             ckpt_list[-1], to_cpu=dist_train, optimizer=optimizer, logger=logger
    #         )
    #         last_epoch = start_epoch + 1

    for name, param in model.named_parameters():
        print(name, '\t',param.requires_grad)

    model.train()  # before wrap to DistributedDataParallel to support fixed some parameters
    if dist_train:
        if cfg.get('MULTI_DB', None):
            model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()],
                                                            broadcast_buffers=False, find_unused_parameters=True)
        else:
            model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()])

    logger.info(model)

    max_len_dataset = len(source_loader)
    total_iters_each_epoch = max_len_dataset if not args.merge_all_iters_to_one_epoch \
                                        else max_len_dataset // args.epochs

    lr_scheduler, lr_warmup_scheduler = build_scheduler(
        optimizer, total_iters_each_epoch=total_iters_each_epoch, total_epochs=args.epochs,
        last_epoch=last_epoch, optim_cfg=cfg.OPTIMIZATION
    )

    # # select proper trainer
    # if cfg.get('MULTI_DB', None):
    #     train_func = train_multi_db_model
    # else:
    #     train_func = train_model
    train_func = train_model


    # -----------------------start training---------------------------
    logger.info('**********************Start training %s/%s(%s)**********************'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))
    train_func(
        model,
        optimizer,
        source_loader,
        source_loader_1 if cfg.get('DATA_CONFIG_SRC_1', None) else None, # added loader
        source_loader_2 if cfg.get('DATA_CONFIG_SRC_2', None) else None,
        model_func=model_fn_decorator(),
        lr_scheduler=lr_scheduler,
        optim_cfg=cfg.OPTIMIZATION,
        start_epoch=start_epoch,
        total_epochs=args.epochs,
        start_iter=it,
        rank=cfg.LOCAL_RANK,
        tb_log=tb_log,
        ckpt_save_dir=ckpt_dir,
        ps_label_dir=ps_label_dir,
        source_sampler=source_sampler,
        target_sampler=source_sampler_2 if cfg.get('DATA_CONFIG_SRC_2', None) else None,
        lr_warmup_scheduler=lr_warmup_scheduler,
        ckpt_save_interval=args.ckpt_save_interval,
        max_ckpt_save_num=args.max_ckpt_save_num,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        logger=logger,
        ema_model=None,
        iter_num_per_target_batch = args.iter_num_per_tar if args.TTT_mode == 0 else 1,
        TTT_mode = args.TTT_mode,
        result_dir = eval_output_dir,
        dist_test = dist_train,
        cfg = cfg,
    )
    # train_model(
    #     model,
    #     optimizer,
    #     source_loader,
    #     model_func=model_fn_decorator(),
    #     lr_scheduler=lr_scheduler,
    #     optim_cfg=cfg.OPTIMIZATION,
    #     start_epoch=start_epoch,
    #     total_epochs=args.epochs,
    #     start_iter=it,
    #     rank=cfg.LOCAL_RANK,
    #     tb_log=tb_log,
    #     ckpt_save_dir=ckpt_dir,
    #     source_sampler=source_sampler,
    #     lr_warmup_scheduler=lr_warmup_scheduler,
    #     ckpt_save_interval=args.ckpt_save_interval,
    #     max_ckpt_save_num=args.max_ckpt_save_num,
    #     merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch
    # )

    if cfg.get('MULTI_DB', None):
        if hasattr(source_set, 'use_shared_memory') and source_set.use_shared_memory:
            source_set.clean_shared_memory()
            source_set_2.clean_shared_memory()
    else:
        if hasattr(source_set, 'use_shared_memory') and source_set.use_shared_memory:
            source_set.clean_shared_memory()

    logger.info('**********************End training %s/%s(%s)**********************\n\n\n'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))


    logger.info('*************** EVALUATION START *****************')
    f = open(eval_output_dir/'result.pkl', 'rb')
    det_annos = pickle.load(f)
    f.close()

    class_names = source_set.class_names
    result_str, result_dict = source_set.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=None
    )

    logger.info(result_str)
    logger.info('*************** EVALUATION END *****************')
    return

if __name__ == '__main__':
    main()