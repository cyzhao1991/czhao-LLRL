for ((i=0; i<25; i++))
do
	python mtl.py --exp_num $((i+25)) --gpu_num 1
	echo "experiment $((i+25)) done." >> log.txt
done