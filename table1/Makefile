all: table1.txt

table1_baseline_column.npz: create_column_data.py
	python3 create_column_data.py baseline $@

table1_baseline_column.txt: create_table_column.py table1_baseline_column.npz
	echo "Runs python3 create_table_column.py table1_baseline_column.npz table1_baseline_column.txt"
	python3 create_table_column.py table1_baseline_column.npz table1_baseline_column.txt

table1.txt: table1_baseline_column.txt
	rm -f table1.txt
	cat table1_baseline_column.txt >> table1.txt

clean: 
	rm table1.txt
