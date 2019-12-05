export CUDA_VISIBLE_DEVICES=5
python evaluate_result.py eval-front mnist.train mnist.test checkpoint_nsga_test_haklnv.pkl 2>eval.err 1>eval.log

