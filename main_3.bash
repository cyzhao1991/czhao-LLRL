for ((i=0; i<25; i++))
do
	python mtl.py --exp_num $((i+50)) --gpu_num 2
	echo "experiment $((i+50)) done." >> log.txt
done