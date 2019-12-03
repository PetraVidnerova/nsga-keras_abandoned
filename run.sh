export CUDA_VISIBLE_DEVICES=0,1
python main.py --trainset mnist.train --testset mnist.test --id test_haklnv 1> test_haklnv.log 2> err_haklnv.log 

