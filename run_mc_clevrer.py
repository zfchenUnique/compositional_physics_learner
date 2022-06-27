"""
Run symbolic reasoning on multiple-choice questions
"""
import os
import json
from tqdm import tqdm
import argparse

from executor import Executor
from simulation import Simulation
import pdb
import numpy as np
EPS = 0.000001

parser = argparse.ArgumentParser()
parser.add_argument('--use_event_ann', default=1, type=int)
parser.add_argument('--use_in', default=0, type=int)  # Use interaction network
parser.add_argument('--program_path', default='/home/zfchen/code/clevrer_dataset_generation_v2/clevrer_question_generation/output/questions_v13_2/open_end_questions.json')
parser.add_argument('--question_path', default='/home/zfchen/code/clevrer_dataset_generation_v2/clevrer_question_generation/output/questions_v13_2/open_end_questions.json')
parser.add_argument('--gt_flag', default=0, type=int)
parser.add_argument('--ann_dir', default='/home/zfchen/code/output/render_output/causal_v13')
parser.add_argument('--track_dir', default='/home/zfchen/code/output/render_output/causal_v13_coco_ann')
parser.add_argument('--raw_motion_prediction_dir', default='')
parser.add_argument('--frame_diff', default=5, type=int)
parser.add_argument('--mc_flag', default=1, type=int)
parser.add_argument('--start_id', default=0, type=int)
parser.add_argument('--num_sim', default=5000, type=int)
parser.add_argument('--file_idx_offset', default=10000, type=int)
parser.add_argument('--result_out_fn', default = 'result_split1.json')
args = parser.parse_args()

question_path = args.question_path
program_path = args.program_path 

print(question_path)
print(program_path)

with open(program_path) as f:
    parsed_pgs = json.load(f)
with open(question_path) as f:
    anns = json.load(f)

total, correct = 0, 0
total_per_q, correct_per_q = 0, 0
total_expl, correct_expl = 0, 0
total_expl_per_q, correct_expl_per_q = 0, 0
total_pred, correct_pred = 0, 0
total_pred_per_q, correct_pred_per_q = 0, 0
total_coun, correct_coun = 0, 0
total_coun_per_q, correct_coun_per_q = 0, 0

total_coun_mass, correct_coun_mass = 0, 0
total_coun_per_q_mass, correct_coun_per_q_mass = 0, 0
total_coun_charge, correct_coun_charge = 0, 0
total_coun_per_q_charge, correct_coun_per_q_charge = 0, 0

pred_map = {'yes': 'correct', 'no': 'wrong', 'error': 'error'}
pbar = tqdm(range(args.num_sim))
prediction_dict = {}

for ann_idx in pbar:
    file_idx = ann_idx + args.start_id 
    question_scene = anns[file_idx-args.file_idx_offset]
    
    sim = Simulation(args, file_idx, n_vis_frames=120, use_event_ann=(args.use_event_ann != 0))
    tmp_pred = {}
    exe = Executor(sim)
    valid_q_idx = 0
    for q_idx, q in enumerate(question_scene['questions']):
        #print('%d\n'%q_idx)
        question = q['question']
        if 'question_type' not in q:
            continue
        q_type = q['question_type']
        q_ann = parsed_pgs[file_idx-args.file_idx_offset]['questions'][q_idx]
        correct_question = True
        pred_list = []
        if 'choices' in q_ann:
            for c in q_ann['choices']:
                full_pg = c['program'] + q_ann['program']
                ans = c['answer'] if 'answer' in c else None
                pred = exe.run(full_pg, debug=False)
                pred = pred_map[pred]
                if ans is not None and  ans == pred:
                    correct += 1
                else:
                    correct_question = False
                total += 1
                
                if q['question_type'].startswith('predictive'):
                    if ans is not None and  ans == pred:
                        correct_pred += 1
                    total_pred += 1

                if q['question_type'].startswith('counterfactual'):
                    if ans is not None and  ans == pred:
                        correct_coun += 1
                    total_coun += 1
                pred_list.append(pred)
        ques_id = q['question_id']
        tmp_pred[ques_id] = {'choices': pred_list, 'question_type': q['question_type'] }

        if correct_question:
            correct_per_q += 1
        total_per_q += 1

        if q['question_type'].startswith('predictive'):
            if correct_question:
                correct_pred_per_q += 1
            total_pred_per_q += 1

        if q['question_type'].startswith('counterfactual'):
            if correct_question:
                correct_coun_per_q += 1
            total_coun_per_q += 1
        valid_q_idx += 1
    prediction_dict[file_idx] =  tmp_pred
    pbar.set_description('per choice {:f}, per questions {:f}'.format(float(correct)*100/max(total, EPS), float(correct_per_q)*100/max(total_per_q, EPS)))

if ans is not None:
    print('============ results ============')
    print('overall accuracy per option: %f %%' % (float(correct) * 100.0 / total))
    print('overall accuracy per question: %f %%' % (float(correct_per_q) * 100.0 / total_per_q))
    print('predictive accuracy per option: %f %%' % (float(correct_pred) * 100.0 / max( total_pred, EPS)))
    print('predictive accuracy per question: %f %%' % (float(correct_pred_per_q) * 100.0 / max(total_pred_per_q, EPS)))
    print('counterfactual accuracy per option: %f %%' % (float(correct_coun) * 100.0 / max(total_coun, EPS)))
    print('counterfactual accuracy per question: %f %%' % (float(correct_coun_per_q) * 100.0 / max(total_coun_per_q, EPS)))
    print('============ results ============')

output_file = args.result_out_fn
if output_file!='':
    with open(output_file, 'w') as fout:
        json.dump(prediction_dict, fout)
