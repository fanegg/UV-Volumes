'''CMU-p4s6'''
python3 run.py --type evaluate \
--cfg_file configs/cmu_exp/p4s6.yaml \
exp_name p4s6 use_lpips True test.frame_sampler_interval 1 use_nb_mask_at_box True save_img True

'''ZJU-313'''
python3 run.py --type evaluate \
--cfg_file configs/zju_mocap_exp/313.yaml \
exp_name zju313 use_lpips True test.frame_sampler_interval 1 use_nb_mask_at_box True \
save_img True T_threshold 0.75 

