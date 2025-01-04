all: table1.txt

table1_baseline.txt: create_table_column.py
	echo "Runs python3 $< baseline $@"
	python3 $< baseline $@

table1.txt: table1_baseline.txt
	rm table1.txt
	cat table1_baseline.txt >> table1.txt

clean: 
	rm table1.txt
