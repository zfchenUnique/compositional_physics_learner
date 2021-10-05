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
parser.add_argument('--start_id', default=0, type=int)
parser.add_argument('--num_sim', default=5000, type=int)
parser.add_argument('--ann_offset', default=0, type=int)
parser.add_argument('--gt_ann_dir', default='/home/tmp_user/code/output/render_output_vislab3/v16/render')
parser.add_argument('--ambi_ques_list_fn', default = './data/ambi_ques_v16.txt')
parser.add_argument('--gen_ambi_flag', default=0, type=int)
parser.add_argument('--save_prediction_fn', default = '')
args = parser.parse_args()

question_path = args.question_path
program_path = args.program_path 

with open(program_path) as f:
    parsed_pgs = json.load(f)
with open(question_path) as f:
    anns = json.load(f)

if args.gen_ambi_flag:
    fh = open(args.ambi_ques_list_fn, 'w')

total, correct = 0, 0

pbar = tqdm(range(args.num_sim))

acc_monitor = {}
acc_monitor_2 = {}
ans_swap = ''
prediction_dict = {}

for ann_idx in pbar:
    file_idx = ann_idx  + args.start_id
    ques_idx = file_idx - args.ann_offset
    question_scene = anns[ques_idx]
    sim = Simulation(args, file_idx, use_event_ann=(args.use_event_ann != 0))
    exe = Executor(sim)
    tmp_dict = {}
    for q_idx, q in enumerate(question_scene['questions']):
        if 'question_type' in q and 'multiple_choice' in q['question_type']:
            continue
        
        question = q['question']
        parsed_pg = parsed_pgs[ques_idx]['questions'][q_idx]['program']
        q_type = parsed_pg[-1]
        if q_type+'_acc' not in acc_monitor:
            acc_monitor[q_type+'_acc'] = 0
            acc_monitor[q_type+'_total'] = 1
        else:
            acc_monitor[q_type+'_total'] +=1

        q_fam = q['question_family']
        if q_fam+'_acc' not in acc_monitor_2:
            acc_monitor_2[q_fam+'_acc'] = 0
            acc_monitor_2[q_fam+'_total'] = 1
        else:
            acc_monitor_2[q_fam+'_total'] +=1

        pred = exe.run(parsed_pg, debug=False)
        ques_id = q['question_id']
        tmp_dict[ques_id] = pred

        ans = q['answer'] if 'answer' in q else None
        if ans is not None and pred == ans:
            correct += 1
            acc_monitor[q_type+'_acc'] +=1
            acc_monitor_2[q_fam+'_acc'] +=1
        elif ans is not None and 'and' in ans: # for query_both     
            eles = ans.split(' ')
            if len(eles)==3:
                ans_swap  = eles[2] + ' and ' + eles[0]
                if pred == ans_swap:
                    correct +=1
                    acc_monitor[q_type+'_acc'] +=1
                    acc_monitor_2[q_fam+'_acc'] +=1
        total += 1
    pbar.set_description('acc: {:f}%%'.format(float(correct)*100/total))
    prediction_dict[file_idx] = tmp_dict

if ans is not None:
    print_monitor(acc_monitor)
    print_monitor(acc_monitor_2)
    print('overall accuracy per question: %f %%' % (float(correct) * 100.0 / total))
if args.save_prediction_fn !='':
    with open(args.save_prediction_fn, 'w') as fh:
        json.dump(prediction_dict, fh)
    print('Saving predictions to %s\n'%(args.save_prediction_fn))
