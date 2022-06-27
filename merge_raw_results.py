import json
import pdb

def merge_result():
    oe_result_path = './test_oe.json' 
    mc_result_path = './test_mc.json' 
    submit_result_path = './test_submit.json' 
    fh_oe = open(oe_result_path, 'rb')
    fh_mc = open(mc_result_path, 'rb')
    fd_oe = json.load(fh_oe)
    fd_mc = json.load(fh_mc)
    assert len(fd_oe) == len(fd_mc)
    start_id = 10000
    vid_num = 2000
    submit_list = []
    for vid in range(start_id, start_id + vid_num):
        tmp_dict = {'scene_index': vid, 'video_filename': 'sim_%05d.mp4'%(vid)}
        vid_str = str(vid)
        oe_dict = fd_oe[vid_str]
        mc_dict = fd_mc[vid_str]
        tmp_ques_list = []
        oe_ques_key_list = sorted(list(oe_dict.keys()))
        mc_ques_key_list = sorted(list(mc_dict.keys()))
        for q_id in oe_ques_key_list:
            tmp_q_pred = {'question_id': int(q_id), 'question_type': 'Factual', 'answer': oe_dict[q_id]}
            tmp_ques_list.append(tmp_q_pred)
        for q_id in mc_ques_key_list:
            tmp_q_pred = {'question_id': int(q_id), 'question_type': mc_dict[q_id]['question_type']}
            choice_list = []
            for c_id, c_pred in enumerate(mc_dict[q_id]['choices']):
                tmp_choice = {'answer': c_pred, 'choice_id': c_id }
                choice_list.append(tmp_choice)
            tmp_q_pred['choices'] = choice_list
            tmp_ques_list.append(tmp_q_pred)
        tmp_dict['questions'] = tmp_ques_list
        submit_list.append(tmp_dict)
    with open(submit_result_path, 'w') as fh_out:
        json.dump(submit_list, fh_out)
        #pdb.set_trace()

if __name__=='__main__':
    merge_result()
