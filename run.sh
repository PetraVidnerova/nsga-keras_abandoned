export CUDA_VISIBLE_DEVICES=0,1
python main.py --conv --trainset mnist2d.train --testset mnist2d.test --id test_haklnv 1> test_haklnv.log 2> err_haklnv.log 

