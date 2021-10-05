import os
import json
import numpy as np
from utils import utils
import pdb
import json

MOVING_V_TH = 0.25  # threshold above which an object is moving
DIR_ANGLE_TH = 20  # threshold for allowed angle deviation wrt each directions
FRAME_DIFF = 5

DV_TH = 0.003
#DIST_TH = 20
DIST_TH = 68
#DIST_TH = 60
print(DIST_TH)
print(DV_TH)
EPS = 0.000001

class Simulation():

    def __init__(self, args, sim_id, n_vis_frames=125, use_event_ann=True):
        
        objs, preds, edges = utils.load_ann(sim_id, args)
        self.objs = objs
        self.preds = preds
        self.edges = edges # for proposal handling
        self.num_objs = len(self.objs)
        self.sim_id = sim_id
       
        self.args = args
        FRAME_DIFF = args.frame_diff
        self.frame_diff = FRAME_DIFF
        self.n_vis_frames = n_vis_frames
        self.moving_v_th = MOVING_V_TH
        self._init_sim_no_event()
        if use_event_ann:
            self._init_collision_gt()

    def get_what_if_id(self, pred):
        p = pred
        if p['what_if'] == -1:
            return -1
        if 'mass' in p:
            cf_id = str(p['what_if']) +'_mass_' + str(p['mass'])
        if 'charge' in p:
            cf_id = str(p['what_if']) +'_charge_' +str(p['charge'])
        return cf_id

    def get_visible_objs(self):
        #return [o['id'] for o in self.objs]
        if self.args.gt_flag:
            obj_list = [o['id'] for o in self.objs if self.is_visible(o['id'])]
        else:
            obj_list = [o['id'] for o in self.objs]
        return obj_list

    def get_static_attrs(self, obj_idx):
        for o in self.objs:
            if o['id'] == obj_idx:
                attrs = {
                    'color': o['color'],
                    'material': o['material'],
                    'shape': o['shape'],
                }
                return attrs
        raise ValueError('Invalid object index')

    def is_visible(self, obj_idx, frame_idx=None, ann_idx=None, what_if=-1):
        if frame_idx is not None or ann_idx is not None:
            frame_ann = self._get_frame_ann(frame_idx, ann_idx, what_if)
            for o in frame_ann['objects']:
                oid = self._get_obj_idx(o)
                if oid == obj_idx:
                    return True
            return False
        else:
            for i, t in enumerate(self.preds[0]['trajectory']):
                if t['frame_index'] < self.n_vis_frames and \
                   self.is_visible(obj_idx, ann_idx=i, what_if=what_if):
                    return True
            return False

    def is_moving(self, obj_idx, frame_idx=None, ann_idx=None):
        if frame_idx is not None or ann_idx is not None:
            frame_ann = self._get_frame_ann(frame_idx, ann_idx)
            for o in frame_ann['objects']:
                oid = self._get_obj_idx(o)
                if oid == obj_idx:
                    speed = np.linalg.norm([o['vx'], o['vy']])
                    return speed > self.moving_v_th
            print(frame_idx, ann_idx)
            raise ValueError('Invalid object index')
        else:
            for i, t in enumerate(self.preds[0]['trajectory']):
                if t['frame_index'] >= self.n_vis_frames:
                    break
                if self.is_visible(obj_idx, ann_idx=i) and \
                   self.is_moving(obj_idx, ann_idx=i):
                    return True
            return False

    def is_moving_up(self, obj_idx, frame_idx=None, ann_idx=None,
                       angle_half_range=DIR_ANGLE_TH):
        if frame_idx is not None:
            frame_ann = self._get_frame_ann(frame_idx, ann_idx)
            for o in frame_ann['objects']:
                oid = self._get_obj_idx(o)
                if oid == obj_idx:
                    theta = np.arctan(o['vy'] / (o['vx']+EPS)) * 180 / np.pi
                    if o['vx'] < 0:
                        theta += 180
                    return theta > 270 - angle_half_range or \
                           theta < -90 + angle_half_range
        else: 
            valid_frm_list = []
            for f in range(0, self.n_vis_frames, FRAME_DIFF):
                if self.is_visible(obj_idx, f)  and  \
                        self.is_moving(obj_idx, f) and \
                   not self.is_moving_up(
                       obj_idx, f, angle_half_range=angle_half_range):
                    valid_frm_list.append(0)
                else:
                    valid_frm_list.append(1)
            ratio = sum(valid_frm_list) / (len(valid_frm_list)+0.000001)
            if ratio >=0.5:
                return True
            else:
                return False
        raise ValueError('Invalid object index')

    def is_moving_down(self, obj_idx, frame_idx=None, ann_idx=None,
                       angle_half_range=DIR_ANGLE_TH):
        if frame_idx is not None:
            frame_ann = self._get_frame_ann(frame_idx, ann_idx)
            for o in frame_ann['objects']:
                oid = self._get_obj_idx(o)
                if oid == obj_idx:
                    theta = np.arctan(o['vy'] / (o['vx']+EPS)) * 180 / np.pi
                    if o['vx'] < 0:
                        theta += 180
                    return theta > 90 - angle_half_range and \
                           theta < 90 + angle_half_range
        else: 
            valid_frm_list = []
            for f in range(0, self.n_vis_frames, FRAME_DIFF):
                if self.is_visible(obj_idx, f)  and  \
                        self.is_moving(obj_idx, f) and \
                   not self.is_moving_down(
                       obj_idx, f, angle_half_range=angle_half_range):
                    valid_frm_list.append(0)
                else:
                    valid_frm_list.append(1)
            ratio = sum(valid_frm_list) / (len(valid_frm_list)+0.000001)
            if ratio >=0.5:
                return True
            else:
                return False
        raise ValueError('Invalid object index')

    def is_moving_left(self, obj_idx, frame_idx=None, ann_idx=None,
                       angle_half_range=DIR_ANGLE_TH):
        if frame_idx is not None:
            frame_ann = self._get_frame_ann(frame_idx, ann_idx)
            for o in frame_ann['objects']:
                oid = self._get_obj_idx(o)
                if oid == obj_idx:
                    theta = np.arctan(o['vy'] / (o['vx'] + EPS)) * 180 / np.pi
                    if o['vx'] < 0:
                        theta += 180
                    return theta > 180 - angle_half_range and \
                           theta < 180 + angle_half_range
        else: 
            valid_frm_list = []
            for f in range(0, self.n_vis_frames, FRAME_DIFF):
                if self.is_visible(obj_idx, f)  and  \
                        self.is_moving(obj_idx, f) and \
                   not self.is_moving_left(
                       obj_idx, f, angle_half_range=angle_half_range):
                    valid_frm_list.append(0)
                else:
                    valid_frm_list.append(1)
            ratio = sum(valid_frm_list) / (len(valid_frm_list)+0.000001)
            if ratio >=0.5:
                return True
            else:
                return False
        raise ValueError('Invalid object index')

    def is_moving_right(self, obj_idx, frame_idx=None, ann_idx=None,
                       angle_half_range=DIR_ANGLE_TH):
        if frame_idx is not None:
            frame_ann = self._get_frame_ann(frame_idx, ann_idx)
            for o in frame_ann['objects']:
                oid = self._get_obj_idx(o)
                if oid == obj_idx:
                    theta = np.arctan(o['vy'] / (o['vx']+EPS)) * 180 / np.pi
                    if o['vx'] < 0:
                        theta += 180
                    return theta < 0 + angle_half_range and \
                           theta > 0 - angle_half_range
        else: 
            valid_frm_list = []
            for f in range(0, self.n_vis_frames, FRAME_DIFF):
                if self.is_visible(obj_idx, f)  and  \
                        self.is_moving(obj_idx, f) and \
                   not self.is_moving_right(
                       obj_idx, f, angle_half_range=angle_half_range):
                    valid_frm_list.append(0)
                else:
                    valid_frm_list.append(1)
            ratio = sum(valid_frm_list) / (len(valid_frm_list)+0.000001)
            if ratio >=0.5:
                return True
            else:
                return False
        raise ValueError('Invalid object index')

    def _init_sim_no_event(self):
        self.in_out = []
        self.collisions = []
        self.cf_events = {}
        for k, p in enumerate(self.preds):
            what_if_id =  self.get_what_if_id(p)
            for i, t in enumerate(p['trajectory']):
                for o in t['objects']:
                    o['id'] = self._get_obj_idx(o)
                    vxs, vys = [], []
                    if i != 0 and not self.is_visible(o['id'], ann_idx=i-1, what_if=what_if_id):
                        if k == 0:
                            self.in_out.append({'frame': t['frame_index'], 'type': 'in', 'object': [o['id']]})
                    elif i != 0:
                        x_prev, y_prev = self._get_obj_location(o['id'], ann_idx=i-1, what_if=what_if_id)
                        vxs.append((o['x'] - x_prev) / self.frame_diff)
                        vys.append((o['y'] - y_prev) / self.frame_diff)
                    if i != len(p['trajectory']) - 1 and not self.is_visible(o['id'], ann_idx=i+1, what_if = what_if_id):
                        if k == 0:
                            self.in_out.append({'frame': p['trajectory'][i+1]['frame_index'], 'type': 'out', 'object': [o['id']]})
                    elif i != len(p['trajectory']) - 1:
                        x_next, y_next = self._get_obj_location(o['id'], ann_idx=i+1, what_if=what_if_id)
                        vxs.append((x_next - o['x']) / self.frame_diff)
                        vys.append((y_next - o['y']) / self.frame_diff)
                    if len(vxs) != 0:
                        o['vx'] = np.average(vxs)
                        o['vy'] = np.average(vys)
                    else:
                        o['vx'], o['vy'] = 0, 0
            if p['what_if'] == -1:
                #print('debug')
                #self.collisions = self._get_col_proposals_counterfact()
                self.collisions = self._get_col_proposals()
            else:
                self.cf_events[what_if_id] = self._get_col_proposals(what_if_id)

    def _init_collision_gt(self):
        ann_path = os.path.join(self.args.gt_ann_dir, 'causal_sim', 'sim_%05d'%self.sim_id, 'annotations', 'annotation.json')
        with open(ann_path, 'r') as fh:
            ann = json.load(fh)
        self.collisions = []
        for c in ann['collisions']:
            obj1_idx = c['object_idxs'][0]
            obj2_idx = c['object_idxs'][1]
            self.collisions.append({
                    'type': 'collision',
                    'object': [obj1_idx, obj2_idx],
                    'frame': int(c['time'] * 25),
                })

    def _init_sim(self):
        self.in_out = []
        self.collisions = []
        p = self.preds[0]
        for i, t in enumerate(p['trajectory']):
            for o in t['objects']:
                o['id'] = self._get_obj_idx(o)
                vxs, vys = [], []
                if i != 0 and not self.is_visible(o['id'], ann_idx=i-1):
                    self.in_out.append({'frame': t['frame_index'], 'type': 'in', 'object': [o['id']]})
                elif i != 0:
                    x_prev, y_prev = self._get_obj_location(o['id'], ann_idx=i-1)
                    vxs.append((o['x'] - x_prev) / self.frame_diff)
                    vys.append((o['y'] - y_prev) / self.frame_diff)
                if i != len(p['trajectory']) - 1 and not self.is_visible(o['id'], ann_idx=i+1):
                    self.in_out.append({'frame': p['trajectory'][i+1]['frame_index'], 'type': 'out', 'object': [o['id']]})
                elif i != len(p['trajectory']) - 1:
                    x_next, y_next = self._get_obj_location(o['id'], ann_idx=i+1)
                    vxs.append((x_next - o['x']) / self.frame_diff)
                    vys.append((y_next - o['y']) / self.frame_diff)
                if len(vxs) != 0:
                    o['vx'] = np.average(vxs)
                    o['vy'] = np.average(vys)
                else:
                    o['vx'], o['vy'] = 0, 0
        for c in p['collisions']:
            obj1_idx = self._get_obj_idx(c['objects'][0])
            obj2_idx = self._get_obj_idx(c['objects'][1])
            self.collisions.append({
                    'type': 'collision',
                    'object': [obj1_idx, obj2_idx],
                    'frame': c['frame'],
                })

        self.cf_events = {}
        for j in range(1, len(self.preds)):
            assert self.preds[j]['what_if'] != -1
            self.cf_events[self.preds[j]['what_if']] = []
            for c in self.preds[j]['collisions']:
                obj1_idx = self._get_obj_idx(c['objects'][0])
                obj2_idx = self._get_obj_idx(c['objects'][1])
                self.cf_events[self.preds[j]['what_if']].append({
                        'type': 'collision',
                        'object': [obj1_idx, obj2_idx],
                        'frame': c['frame'],
                    })    

    def _get_obj_idx(self, obj):
        for o in self.objs:
            if o['color'] == obj['color'] and \
               o['material'] == obj['material'] and \
               o['shape'] == obj['shape']:
                return o['id']
        return -1

    def search_obj_info_by_id(self, obj_id):
        for idx, obj_info in enumerate(self.objs):
            if obj_info['id']==obj_id:
                return obj_info
        return -1

    def _get_frame_ann(self, frame_idx=None, ann_idx=None, what_if=-1):
        assert ann_idx is not None or frame_idx is not None
        target = None
        if frame_idx is not None:
            for t in self.search_pred_by_cf_id(what_if)['trajectory']:
                if t['frame_index'] == frame_idx:
                    target = t
                    break
        else:
            target = self.search_pred_by_cf_id(what_if)['trajectory'][ann_idx]
        if target is None:
            raise ValueError('Invalid input frame')
        return target

    def _get_obj_location(self, obj_idx, frame_idx=None, ann_idx=None, what_if=-1):
        assert self.is_visible(obj_idx, frame_idx, ann_idx, what_if=what_if)
        frame_ann = self._get_frame_ann(frame_idx, ann_idx, what_if)
        for o in frame_ann['objects']:
            if self._get_obj_idx(o) == obj_idx:
                return o['x'], o['y']

    def search_pred_by_cf_id(self, what_if):
        if what_if==-1:
            return  self.preds[what_if+1]
        obj_id, prop, prop_val = int(what_if.split('_')[0]), what_if.split('_')[1], int(what_if.split('_')[2])
        for pred_id, pred_info in enumerate(self.preds):
            what_if = pred_info['what_if']
            if obj_id!=what_if:
                continue
            if prop in pred_info and pred_info[prop]==prop_val:
                return pred_info
        return 'error'

    def get_trace(self, obj, what_if=-1):
        output = []
        pred = self.search_pred_by_cf_id(what_if)
        for t in pred['trajectory']:
            for o in t['objects']:
                if o['id'] == obj:
                    o['frame'] = t['frame_index']
                    output.append(o)
        return output

    def _get_col_frame_proposals(self, obj, what_if=-1):
        proposed_frames = []
        trace = self.get_trace(obj, what_if)
        dvs = []
        for i, o in enumerate(trace):
            if i > 0:
                dvx = o['vx'] - trace[i-1]['vx']
                dvy = o['vy'] - trace[i-1]['vy']
                dv = np.linalg.norm([dvx, dvy])
            else:
                dv = 0 
            dvs.append(dv)
        for j, dv in enumerate(dvs):
            if j != 0 and j != len(dvs)-1:
                if dv > dvs[j-1] and dv > dvs[j+1] and dv > DV_TH and dv < 5 and self.is_visible(obj, frame_idx=trace[j]['frame'], what_if=what_if):
                    proposed_frames.append(trace[j]['frame'])
        return proposed_frames

    def _get_closest_obj(self, obj, frame_idx, what_if=-1):
        assert self.is_visible(obj, frame_idx=frame_idx, what_if=what_if)
        xo, yo = self._get_obj_location(obj, frame_idx=frame_idx, what_if=what_if)
        obj_idxs = [o['id'] for o in self.objs]
        min_dist = 99999
        closest_obj = -1
        for io in obj_idxs:
            if io != obj and self.is_visible(io, frame_idx=frame_idx, what_if=what_if):
                x, y = self._get_obj_location(io, frame_idx=frame_idx, what_if=what_if)
                dist = np.linalg.norm([x-xo, y-yo])
                if dist < min_dist:
                    min_dist = dist
                    closest_obj = io
        return closest_obj, min_dist

    def _get_closest_obj_list(self, obj, frame_idx, what_if=-1):
        assert self.is_visible(obj, frame_idx=frame_idx, what_if=what_if)
        xo, yo = self._get_obj_location(obj, frame_idx=frame_idx, what_if=what_if)
        obj_idxs = [o['id'] for o in self.objs]
        min_dist = 99999
        closest_obj = -1
        closest_obj_list  = []
        min_dist_list  = []
        for io in obj_idxs:
            if io != obj and self.is_visible(io, frame_idx=frame_idx, what_if=what_if):
                x, y = self._get_obj_location(io, frame_idx=frame_idx, what_if=what_if)
                dist = np.linalg.norm([x-xo, y-yo])
                min_dist_list.append(dist)
                closest_obj_list.append(io)
        return closest_obj_list, min_dist_list

    def _get_col_proposals_counterfact(self, what_if=-1):
        cols = []
        col_pairs = []
        obj_idxs = [o['id'] for o in self.objs]
        for io in obj_idxs:
            col_frames = self._get_col_frame_proposals(io, what_if)
            for f in col_frames:
                partner_list, dist_list = self._get_closest_obj_list(io, f, what_if)
                for partner, dist in zip(partner_list, dist_list):
                    #if what_if==-1:
                    #    print('frame: %d, object indexes: %d %d, dist: %f\n'%(f, io, partner, dist))
                    if dist < DIST_TH and {io, partner} not in col_pairs:
                        col_event = {
                            'type': 'collision',
                            'object': [io, partner],
                            'frame': f,
                        }
                        cols.append(col_event)
                        col_pairs.append({io, partner})
        return cols

    def _get_col_proposals(self, what_if=-1):
        cols = []
        col_pairs = []
        obj_idxs = [o['id'] for o in self.objs]
        for io in obj_idxs:
            col_frames = self._get_col_frame_proposals(io, what_if)
            for f in col_frames:
                partner, dist = self._get_closest_obj(io, f, what_if)
                #if what_if==-1:
                #    print('frame: %d, object indexes: %d %d, dist: %f\n'%(f, io, partner, dist))
                if dist < DIST_TH and {io, partner} not in col_pairs:
                    col_event = {
                        'type': 'collision',
                        'object': [io, partner],
                        'frame': f,
                    }
                    cols.append(col_event)
                    col_pairs.append({io, partner})
        return cols
    
    def is_charged(self, obj_idx):
        return self.objs[obj_idx]['charge']!=0
    
    def is_light(self, obj_idx):
        return self.objs[obj_idx]['mass']==1
