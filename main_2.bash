for ((i=0; i<10; i++))
do
	python trpo_main.py --exp $((i)) --gpu 1
	echo "gpu 1 experiment $((i)) done." >> log.txt
done