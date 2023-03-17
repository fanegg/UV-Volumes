'''CMU-p4s6'''
python3 train_net.py \
--cfg_file configs/cmu_exp/p4s6.yaml \
exp_name p4s6 resume False output_depth True

'''ZJU-313'''
python3 train_net.py \
--cfg_file configs/zju_mocap_exp/313.yaml \
exp_name zju313 resume False output_depth True

'''H36M-s6p'''
python3 train_net.py \
--cfg_file configs/h36m_exp/s6p.yaml \
exp_name s6p resume False output_depth True
