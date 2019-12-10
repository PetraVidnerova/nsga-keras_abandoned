export CUDA_VISIBLE_DEVICES=3
python evaluate_result.py list-front checkpoint_nsga_haklnv_conv_3.pkl 1> eval_3.log
python evaluate_result.py --conv True eval-front mnist2d.train mnist2d.test checkpoint_nsga_haklnv_conv_3.pkl 2>eval_3.err 1>>eval_3.log

