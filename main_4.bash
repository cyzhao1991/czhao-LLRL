for ((i=0; i<10; i++))
do
	python trpo_main.py --exp $((i)) --gpu 3 &
	# echo "gpu 3 experiment $((i)) done." >> log.txt
done