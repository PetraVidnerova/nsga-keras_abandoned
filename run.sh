export CUDA_VISIBLE_DEVICES=0 
python main.py --type conv --trainset mnist2d.train --testset mnist2d.test --id haklnv_conv 1> test_haklnv_conv.log 2> err_haklnv_conv.log 

# for I in `seq 1 4`
# do
#     export CUDA_VISIBLE_DEVICES=$I 
#     python main.py --type conv --trainset mnist2d.train --testset mnist2d.test --id haklnv_conv_$I 1> test_haklnv_conv_$I.log 2> err_haklnv_conv_$I.log &
# done 
