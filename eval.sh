for I in `seq 0 19`
do
 echo "---- $I ---"   
 python evaluate_result.py evaluate $I mnist.train mnist.test checkpoint_nsga_test.pkl 2>/dev/null 
done
