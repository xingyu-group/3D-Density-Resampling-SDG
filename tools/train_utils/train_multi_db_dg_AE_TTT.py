import torch
import os
import glob
import tqdm
from torch.nn.utils import clip_grad_norm_
from pcdet.models import load_data_to_gpu
import copy
import pickle
from pcdet.utils import common_utils
import random
from pcdet.ops.iou3d_nms import iou3d_nms_utils
import copy

def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    # precision
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['precision_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_P_%s' % str(cur_thresh), 0)
        metric['precision_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_P_%s' % str(cur_thresh), 0)
    metric['dt_num'] += ret_dict.get('dt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])


def visualize_boxes_batch(batch):
    import visualize_utils as vis
    import mayavi.mlab as mlab
    for b_idx in range(batch['batch_size']):
        points = batch['points'][batch['points'][:, 0] == b_idx][:, 1:]

        if 'debug' not in batch:
            vis.draw_scenes(points, ref_boxes=batch['gt_boxes'][b_idx, :, :7],
                            scores=batch['scores'][b_idx])
        else:
            vis.draw_scenes(points, ref_boxes=batch['gt_boxes'][b_idx, :, :7],
                            gt_boxes=batch['debug'][b_idx]['gt_boxes_lidar'],
                            scores=batch['scores'][b_idx])
        mlab.show(stop=True)

def merge_two_batch_data(batch_1, batch_2):
    import numpy as np
    ret = {}
    for key, val in batch_1.items():
        if key in ['batch_size']:
            continue
        else:
            ret[key] = np.stack(val, axis=0)
    for key, val in batch_2.items():
        val_cat = []
        if key in ['batch_size']:
            continue
        elif key in ['gt_boxes']:
            assert batch_1[key][0].shape[-1] == val[0].shape[-1]
            max_gt = max([len(x) for x in batch_1[key]]) + max([len(x) for x in val])
            batch_gt_boxes3d = np.zeros((batch_1['batch_size']*2, max_gt, val[0].shape[-1]), dtype=np.float32)
            for k in range(batch_1['batch_size']):
                batch_gt_boxes3d[k, :batch_1[key][k].__len__(), :] = batch_1[key][k]
            for k in range(batch_2['batch_size']):
                batch_gt_boxes3d[k+batch_1['batch_size'], :val[k].__len__(), :] = val[k]
            ret[key] = batch_gt_boxes3d
        else:
            val_cat.append(batch_1[key])
            val_cat.append(val)
            ret[key] = np.concatenate(val_cat, axis=0)
            #ret[key] = np.stack(val, axis=0)
    ret['batch_size'] = batch_1['batch_size']*2
    return ret

        
def train_one_epoch_multi_db(model, optimizer, train_loader_1, train_loader_src_1, train_loader_src_2, model_func, lr_scheduler, accumulated_iter, optim_cfg, 
                    rank, tbar, total_it_each_epoch, dataloader_iter_1, dataloader_iter_src_1, dataloader_iter_src_2, ckpt_save_dir, dist_test, cfg, TTT_mode, 
                    tb_log=None, leave_pbar=False, iter_num_per_target_batch=None, result_dir=None, logger=None):

    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
        'dt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0
        # precision
        metric['precision_roi_%s' % str(cur_thresh)] = 0
        metric['precision_rcnn_%s' % str(cur_thresh)] = 0

    all_sample_dir = final_output_dir / 'all_samples'
    all_sample_dir.mkdir(parents=True, exist_ok=True)

    dataset_target = train_loader_1.dataset
    class_names = dataset_target.class_names
    det_annos = []

    state_pretrained_model = copy.deepcopy(model.state_dict()) 


    if total_it_each_epoch == len(train_loader_1):
        dataloader_iter_1 = iter(train_loader_1)

    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train', dynamic_ncols=True)

    for cur_it in range(total_it_each_epoch):      
        try:
            batch_target = next(dataloader_iter_1)
        except StopIteration:
            dataloader_iter_1 = iter(train_loader_1)
            batch_target = next(dataloader_iter_1)
        
        lr_scheduler.step(accumulated_iter)

        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr'] 

        if tb_log is not None:
            tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
                
        # accumulated_iter += 1

        load_data_to_gpu(batch_target)

        
        for cur_it_TT in range(iter_num_per_target_batch):
        
            model.train()
            optimizer.zero_grad()

            ret_dict_s, tb_dict_s, disp_dict_s = model(batch_target)

            loss = 0
            loss_disp_dict = {}
            for loss_item in list(ret_dict_s['loss'].keys()):
                loss += ret_dict_s['loss'][loss_item].mean()
                loss_disp_dict[loss_item] = ret_dict_s['loss'][loss_item].item()

            loss.backward()
            clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
            optimizer.step()         

            disp_dict_s.update(loss_disp_dict)
            disp_dict_s.update({'lr': cur_lr})

            # log to console and tensorboard
            # save the log of the source domain ONE
            if rank == 0:
                tbar.set_postfix(disp_dict_s)
                tbar.refresh()

                if tb_log is not None:
                    tb_log.add_scalar('train/loss', loss, accumulated_iter)
                    tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
                    for key, val in tb_dict_s.items():
                        tb_log.add_scalar('train/' + key, val, accumulated_iter)


        if rank == 0:
                pbar.update()
                pbar.set_postfix(dict(total_it=accumulated_iter))

        # test-time testing

        batch_target['points'] = batch_target['points_org']
        model.eval()
        with torch.no_grad():
            pred_dicts, ret_dict = model(batch_target)
        disp_dict = {}

        statistics_info(cfg, ret_dict, metric, disp_dict)

        annos = dataset_target.generate_prediction_dicts(
            batch_target, pred_dicts, class_names,
            output_path=final_output_dir
        )


        # annos_ID = ''
        # for frame_id in batch_target['frame_id']:
        #     annos_ID += frame_id + '_&_'
        # annos_ID = annos_ID[:-3]
        # # print(cur_it, annos_ID)
        # with open(all_sample_dir /(annos_ID +'.pkl'), 'wb') as f:
        #     pickle.dump(annos, f)
        
        det_annos += annos

        if TTT_mode == 0:
            model.load_state_dict(state_pretrained_model)  # standard


    if rank == 0:
        pbar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset_target), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')
    
    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    dt_num_cnt = metric['dt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_precision = metric['precision_roi_%s' % str(cur_thresh)] / max(dt_num_cnt, 1)
        cur_rcnn_precision = metric['precision_rcnn_%s' % str(cur_thresh)] / max(dt_num_cnt, 1)
        logger.info('precision_roi_%s: %f' % (cur_thresh, cur_roi_precision))
        logger.info('precision_rcnn_%s: %f' % (cur_thresh, cur_rcnn_precision))
        ret_dict['precision/roi_%s' % str(cur_thresh)] = cur_roi_precision
        ret_dict['precision/rcnn_%s' % str(cur_thresh)] = cur_rcnn_precision

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)

    return accumulated_iter


def train_multi_db_model(model, optimizer, train_src_loader, train_src_loader_1, train_src_loader_2, model_func, lr_scheduler, optim_cfg,
                start_epoch, total_epochs, start_iter, rank, tb_log, ckpt_save_dir, ps_label_dir, dist_test, cfg, TTT_mode,
                source_sampler=None, target_sampler=None, lr_warmup_scheduler=None, ckpt_save_interval=1,
                max_ckpt_save_num=50, merge_all_iters_to_one_epoch=False, logger=None, ema_model=None, result_dir=None, iter_num_per_target_batch=None):
    accumulated_iter = start_iter
    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
        total_it_each_epoch = len(train_src_loader)

        if merge_all_iters_to_one_epoch:
            raise NotImplementedError

        dataloader_iter_1 = iter(train_src_loader)
        dataloader_iter_src_1 = iter(train_src_loader_1) if train_src_loader_1 is not None else None
        dataloader_iter_src_2 = iter(train_src_loader_2) if train_src_loader_2 is not None else None
        for cur_epoch in tbar:
            if source_sampler is not None:
                source_sampler.set_epoch(cur_epoch)

            # train one epoch
            if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
                cur_scheduler = lr_warmup_scheduler
            else:
                cur_scheduler = lr_scheduler
            accumulated_iter = train_one_epoch_multi_db(
                model, optimizer, 
                train_src_loader, train_src_loader_1, train_src_loader_2, 
                model_func,
                lr_scheduler=cur_scheduler,
                accumulated_iter=accumulated_iter, optim_cfg=optim_cfg,
                rank=rank, tbar=tbar, tb_log=tb_log,
                leave_pbar=(cur_epoch + 1 == total_epochs),
                total_it_each_epoch=total_it_each_epoch,
                dataloader_iter_1=dataloader_iter_1,
                dataloader_iter_src_1=dataloader_iter_src_1,
                dataloader_iter_src_2=dataloader_iter_src_2,
                ckpt_save_dir=ckpt_save_dir,
                result_dir = result_dir,
                iter_num_per_target_batch = iter_num_per_target_batch,
                dist_test = dist_test,
                cfg = cfg,
                logger = logger,
                TTT_mode = TTT_mode,
            )



def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def checkpoint_state(model=None, optimizer=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    try:
        import pcdet
        version = 'pcdet+' + pcdet.__version__
    except:
        version = 'none'

    return {'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state': optim_state, 'version': version}


def save_checkpoint(state, filename='checkpoint'):
    if False and 'optimizer_state' in state:
        optimizer_state = state['optimizer_state']
        state.pop('optimizer_state', None)
        optimizer_filename = '{}_optim.pth'.format(filename)
        torch.save({'optimizer_state': optimizer_state}, optimizer_filename)

    filename = '{}.pth'.format(filename)
    torch.save(state, filename)