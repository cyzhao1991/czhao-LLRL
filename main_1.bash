for ((i=0; i<25; i++))
do
	python mtl.py --exp_num $((i)) --gpu_num 0
	echo "experiment $((i)) done." >> log.txt
done