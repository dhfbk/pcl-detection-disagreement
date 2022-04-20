import os

# Paths
DATA_FOLDER = "data"
GOLD_FOLDER = os.path.join(DATA_FOLDER, "dev_gold")
DEV_FILEPATH = os.path.join(DATA_FOLDER, "paragraphs.dev")

if not os.path.exists(GOLD_FOLDER): os.makedirs(GOLD_FOLDER)
output_file_1 = open(os.path.join(GOLD_FOLDER, "task1.txt"), "w")
output_file_2 = open(os.path.join(GOLD_FOLDER, "task2.txt"), "w")

with open(DEV_FILEPATH, "r") as f:
	for line in f:
		parts = line.rstrip().split("\t")
		label_1 = parts[6]
		label_2 = ",".join(parts[7:14])
		output_file_1.write(str(label_1) + "\n")
		output_file_2.write(str(label_2) + "\n")

output_file_1.close()
output_file_2.close()