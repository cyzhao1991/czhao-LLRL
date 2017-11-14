for ((i=0; i<100; i++))
do
	python stl.py --exp_num $((i))
	# echo "experiment $((i)) done." >> log.txt
done