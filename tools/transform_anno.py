import os
import pdb
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default=0, type=int, help='0 for GT parsing' )
parser.add_argument('--st_id', default=0, type=int, help='start id' )
parser.add_argument('--ed_id', default=1000, type=int, help='end id' )
parser.add_argument('--gt_ques_ann_dir', default='/home/tmp_user/code/clevrer_dataset_generation_v2/clevrer_question_generation/output/questions_v13')

def prepare_gt_programs(args):
    for vid in range(args.st_id, args.ed_id):
        sim_str = 'sim_%05d'%vid
        gt_oe_ques_path = os.path.join(args.gt_ques_ann_dir, 'open_end_questions.json')
        with open(gt_oe_ques_path, 'r') as fh:
            gt_oe_ques_info = json.load(fh)
        gt_mc_ques_path = os.path.join(args.gt_ques_ann_dir, 'multiple_choice_questions.json')
        with open(gt_mc_ques_path, 'r') as fh:
            gt_mc_ques_info = json.load(fh)
        assert gt_oe_ques_info['scene_index'] == gt_mc_ques_info['scene_index']
        out_dict = {"scene_index": gt_oe_ques_info['scene_index'],
                "video_filename": gt_oe_ques_info['video_filename']}

if __name__=='__main__':
    args = parser.parse_args()
    if args.mode==0:
        prepare_gt_programs(args)

