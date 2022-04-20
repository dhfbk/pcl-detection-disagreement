import argparse
import os

DATA_FOLDER = "data"
GOLD_FOLDER = os.path.join(DATA_FOLDER, "gold-dev")

POSITIVE_LABEL = "1"
NEGATIVE_LABEL = "0"

# input file: PCLlabels.dev.out


def create_shared_task_output_files(input_filepath, output_folder, tasks):
	"""A function that converts predictions to the shared task output format."""

	if not os.path.exists(output_folder): os.makedirs(output_folder)

	# If predictions are for task1 (or all), create the task1 output file
	if (tasks == "all") or (tasks == "task1"):
		task1_output = open(os.path.join(output_folder, "task1.txt"), "w")

		with open(input_filepath) as f:
			for line in f:
				label = line.rstrip().split("\t")[6]
				task1_output.write(str(label) + "\n")
		task1_output.close()

	# If predictions are for task2 (or all), create the task2 output file
	# In case of "all", an additional file with inferred predictions for task1 will be created
	if (tasks == "all") or (tasks == "task2"):
		task2_output = open(os.path.join(output_folder, "task2.txt"), "w")
		task1_from_task2_output = open(os.path.join(output_folder, "task1.txt"), "w")

		with open(input_filepath) as f:
			for line in f:
				labels = line.rstrip().split("\t")[7:14]
				labels_formatted = ",".join(labels)
				task2_output.write(str(labels_formatted) + "\n")
				if POSITIVE_LABEL in labels:
					task1_from_task2_output.write(POSITIVE_LABEL + "\n")
				else:
					task1_from_task2_output.write(NEGATIVE_LABEL + "\n")
		task2_output.close()
		task1_from_task2_output.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-I", "--input_filepath", type=str, required=True, 
        help="The filepath of predictions on the dev set (i.e., path to the 'PARlabels.dev.out' file).")
    parser.add_argument("-O", "--output_folder", type=str, required=True, 
        help="The folder where to write predictions in a numeric format for evaluation.")
    parser.add_argument("-T", "--tasks", type=str, required=True, 
        default="all", choices=["all", "task1", "task2"], 
        help="The task tackled by the model which created the prediction file.")
    args = parser.parse_args()

    # Converts predictions to the shared task output format
    create_shared_task_output_files(args.input_filepath, args.output_folder, args.tasks)