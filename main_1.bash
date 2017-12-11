for ((i=0; i<10; i++))
do
	python trpo_main.py --exp $((i)) --gpu 0
	echo "gpu 0 experiment $((i)) done." >> log.txt
done