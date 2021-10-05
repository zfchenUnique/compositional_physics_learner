import os
import pdb
import argparse
import json
import pycocotools.mask as coco_mask
import numpy as np

def get_obj_id_by_attr(obj_list, obj):
    for obj_info in obj_list:
        if obj_info['color'] == obj['color'] and obj_info['shape']==obj['shape'] \
                and obj_info['material'] == obj['material']:
            return obj_info['id']
    return -1

def get_centre_from_mask(mask):
    [x, y, w, h] = coco_mask.toBbox(mask)
    return x+w*0.5, y+h*0.5

def load_gt_ann(sim_id, args):
    sim_str = 'sim_%05d'%sim_id
    ann_path = os.path.join(args.ann_dir, sim_str, 'annotations', 'annotation.json')
    with open(ann_path, 'r') as fh:
        ann = json.load(fh)
    objs = []
    for obj_id, obj_info in enumerate(ann['config']):
        tmp_obj = {'color': obj_info['color'],
                'material': obj_info['material'],
                'shape': obj_info['shape'],
                'mass': obj_info['mass'],
                'charge': obj_info['charge'],
                'id': obj_id}
        objs.append(tmp_obj)
    ann_mask_path  = os.path.join(args.track_dir, sim_str+'.json')
    with open(ann_mask_path, 'r') as fh:
        ann_mask = json.load(fh)['scenes']
    scene_num = len(ann_mask)
    preds = []
    tmp_pred = []
    for frm_id  in range(0, scene_num, args.frame_diff):
        frm_obj_info = ann_mask[frm_id]['objects'] 
        out_frm_info = {'frame_index': frm_id, 'objects':[]}
        obj_list = []
        for obj_id, obj_info in enumerate(frm_obj_info): 
            tmp_obj = {'color': obj_info['color'],
                    'material': obj_info['material'],
                    'shape': obj_info['shape'],
                    'frame': frm_id}
            obj_id = get_obj_id_by_attr(objs, obj_info) 
            tmp_obj['id'] = obj_id
            x, y = get_centre_from_mask( obj_info['mask'])
            tmp_obj['x'] = x
            tmp_obj['y'] = y
            obj_list.append(tmp_obj)
        out_frm_info['objects'] = obj_list
        tmp_pred.append(out_frm_info)
    preds = [{'what_if': -1, 'trajectory': tmp_pred, 'collisions': ann['collisions'] }]
    return objs, preds

def load_mc_ann(sim_id, args):
    IMG_H, IMG_W = 320, 480
    sim_str = 'sim_%05d'%sim_id
    if args.gt_flag:
        ann_path = os.path.join(args.ann_dir, sim_str, 'annotations', 'annotation.json')
    else:
        ann_path = os.path.join(args.ann_dir, sim_str + '.json')
        
    with open(ann_path, 'r') as fh:
        ann = json.load(fh)
    objs = []
    motion_path = os.path.join(args.raw_motion_prediction_dir, sim_str+'.json') 
    with open(motion_path, 'r') as fh:
        raw_pred = json.load(fh) 
    for obj_id, obj_info in enumerate(ann['config']):
        tmp_obj = {'color': obj_info['color'],
                'material': obj_info['material'],
                'shape': obj_info['shape'],
                'id': obj_id}
        if 'mass' in obj_info:        
            tmp_obj['mass'] = obj_info['mass']
        else:
            tmp_obj['mass'] = 1
        if 'charge' in obj_info:
            tmp_obj['charge'] =  obj_info['charge']
        else:
            tmp_obj['charge'] =  0
        objs.append(tmp_obj)
    raw_pred_list = []
    for key_id in ['future', 'mass', 'charge']:
        pred_tmp  = raw_pred[key_id]
        if isinstance(pred_tmp, dict):
            raw_pred_list.append(pred_tmp)
        elif isinstance(pred_tmp, list):
            raw_pred_list += pred_tmp 
    preds = []
    for pred_id, pred_tmp in enumerate(raw_pred_list):
        tmp_output = {"what_if": pred_tmp["what_if"]}
        if "mass" in pred_tmp:
            tmp_output['mass'] = pred_tmp['mass']
        if "charge" in pred_tmp:
            tmp_output['charge'] = pred_tmp['charge']
        track = pred_tmp['trajectories']
        assert len(track) == len(ann['config'])
        obj_num, frm_num = len(track), len(track[0])
        track_list = [] 
        for frm_id in range(frm_num):
            frm = frm_id * args.frame_diff
            frm_info = {'frame_index': frm, "objects": []}
            objs_list = []
            for obj_id in range(obj_num):
                x_c, y_c, w, h = track[obj_id][frm_id]
                if x_c <0 or x_c>1 or y_c<0 or y_c>1:
                    continue
                obj_info = {"color": ann['config'][obj_id]["color"],
                        "shape": ann['config'][obj_id]["shape"],
                        "material": ann['config'][obj_id]["material"],
                        "frame": frm, 
                        "x": x_c * IMG_W, 
                        "y": y_c * IMG_H,
                        "w": w * IMG_W,
                        "h": h * IMG_H,
                        "id": obj_id}
                objs_list.append(obj_info)
            frm_info["objects"] = objs_list
            track_list.append(frm_info)
        tmp_output["trajectory"] = track_list
        preds.append(tmp_output)
    if not args.gt_flag and 'edegs' in ann:
        edges = np.array(ann['edges'])
        assert len(objs)==edges.shape[0], 'Shape inconsistent'
    else:
        edges = None
    return objs, preds, edges

def load_ann(sim_id, args):
    return load_mc_ann(sim_id, args)

def print_monitor(monitor):
    for key_id, acc_num in monitor.items():
        if key_id.endswith('total'):
            continue
        q_type = key_id.rsplit('_', 1)[0]
        acc = acc_num /( 1.0 * monitor[q_type +'_total'] )
        print('%s: acc: %f, %d/%d\n'%(q_type, acc, acc_num, monitor[q_type+'_total']))
if __name__=='__main__':
    pdb.set_trace()
