"""
Run symbolic reasoning on open-ended questions
"""
import os
import json
from tqdm import tqdm
import argparse

from executor import Executor
from simulation import Simulation
import pdb
from utils.utils import print_monitor
import numpy as np
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--use_event_ann', default=1, type=int)
parser.add_argument('--use_in', default=0, type=int)  # Interaction network
parser.add_argument('--program_path', default='/home/tmp_user/code/clevrer_dataset_generation_v2/clevrer_question_generation/output/questions_v13_2/open_end_questions.json')
parser.add_argument('--question_path', default='/home/tmp_user/code/clevrer_dataset_generation_v2/clevrer_question_generation/output/questions_v13_2/open_end_questions.json')
parser.add_argument('--gt_flag', default=0, type=int)
parser.add_argument('--ann_dir', default='/home/tmp_user/code/output/render_output/causal_v13')
parser.add_argument('--track_dir', default='/home/tmp_user/code/output/render_output/causal_v13_coco_ann')
parser.add_argument('--frame_diff', default=5, type=int)
parser.add_argument('--mc_flag', default=0, type=int)
parser.add_argument('--raw_motion_prediction_dir', default='')
parser.add_argument('--invalid_video_fn', default = 'invalid_video_v14.txt')
parser.add_argument('--start_id', default=0, type=int)
parser.add_argument('--num_sim', default=5000, type=int)
args = parser.parse_args()

question_path = args.question_path
program_path = args.program_path 

with open(program_path) as f:
    parsed_pgs = json.load(f)
with open(question_path) as f:
    anns = json.load(f)

total, correct = 0, 0

pbar = tqdm(range(args.num_sim))

acc_monitor = {}
ans_swap = ''

if os.path.isfile(args.invalid_video_fn):
    print("%s exists."%invalid_list)
    sys.exit()
else:
    fh = open(args.invalid_video_fn, 'w')
invalid_num = 0
for ann_idx in pbar:
    file_idx = ann_idx  + args.start_id
    question_scene = anns[file_idx]
    sim = Simulation(args, file_idx, use_event_ann=(args.use_event_ann != 0))
    if len(sim.objs)!=len(sim.get_visible_objs()):
        fh.write('%d\n'%file_idx)
        fh.flush()
        invalid_num +=1
print('Invalid file num: %d\n'%invalid_num)
fh.close()
