QUES_PATH='/home/tmp_user/code/output/comPhy/questions/test.json'
ANN_DIR='/home/tmp_user/code/output/render_output_vislab3/v16_test/prediction_prp_mass_charge'
#PRED_DIR='/home/tmp_user/code/output/render_output_vislab3/v16_test/predictions_motion_prp_no_ref_v2'
PRED_DIR='/home/tmp_user/code/output/render_output_vislab3/v16_test/predictions_motion_prp_no_ref_v2_vis50_new'
python run_mc_clevrer.py \
    --gt_flag 0 \
    --use_event_ann 0 \
    --raw_motion_prediction_dir ${PRED_DIR} \
    --ann_dir ${ANN_DIR} \
    --start_id 10000 \
    --num_sim 2000 \
    --program_path  ${QUES_PATH} \
    --question_path ${QUES_PATH} \
    --result_out_fn test_mc.json \
    #--program_path /home/tmp_user/code/clevrer_dataset_generation_v2/clevrer_question_generation/output/questions_v16_3_1_test/multiple_choice_questions.json \
    #--question_path /home/tmp_user/code/clevrer_dataset_generation_v2/clevrer_question_generation/output/questions_v16_3_1_test/multiple_choice_questions.json \
