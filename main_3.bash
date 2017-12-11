for ((i=0; i<10; i++))
do
	python trpo_main.py --exp $((i)) --gpu 2 &
	# echo "gpu 2 experiment $((i)) done." >> log.txt
done