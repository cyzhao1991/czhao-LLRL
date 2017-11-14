for ((i=0; i<=25; i++))
do
	python mtl.py --exp_num $((i+75)) --gpu_num 3
	echo "experiment $((i+75)) done." >> log.txt
done