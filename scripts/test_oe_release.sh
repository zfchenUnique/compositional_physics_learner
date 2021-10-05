QUES_PATH='/home/tmp_user/code/output/comPhy/questions/test.json'
ANN_DIR='/home/tmp_user/code/output/render_output_vislab3/v16_test/prediction_prp_mass_charge'
PRED_DIR='/home/tmp_user/code/output/render_output_vislab3/v16_test/predictions_motion_prp_no_ref_v2_vis50_new'
#PRED_DIR='/home/tmp_user/code/output/render_output_vislab3/v16_test/predictions_motion_prp_no_ref_v2'
python run_oe_clevrer.py \
    --gt_flag 0 \
    --use_event_ann 0 \
    --program_path ${QUES_PATH} \
    --question_path ${QUES_PATH} \
    --raw_motion_prediction_dir ${PRED_DIR} \
    --ann_dir ${ANN_DIR} \
    --start_id 10000 \
    --num_sim 50 \
    --ann_offset 10000 \
    --save_prediction_fn test_oe.json \
