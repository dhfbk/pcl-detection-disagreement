import os
import pandas as pd
import sys

# Paths
DATA_FOLDER = "data"
DATA_RAW_FOLDER = os.path.join(DATA_FOLDER, "raw")
DATA_FILEPATH_TASK1 = os.path.join(DATA_RAW_FOLDER, "dontpatronizeme_pcl.tsv")
DATA_FILEPATH_TASK2 = os.path.join(DATA_RAW_FOLDER, "dontpatronizeme_categories.tsv")
PRACTICE_TRAIN_FILEPATH = os.path.join(DATA_RAW_FOLDER, "train_semeval_parids-labels.csv")
PRACTICE_DEV_FILEPATH = os.path.join(DATA_RAW_FOLDER, "dev_semeval_parids-labels.csv")
TEST_FILEPATH = os.path.join(DATA_RAW_FOLDER, "task4_test.tsv")

# Data headers and formats (separators, lines to be skipped, additional task2 cols)
HEADER_TASK1 = ["par_id", "art_id", "keyword", "country_code", "text", "label"]
HEADER_TASK2 = ["par_id", "art_id", "text", "keyword", "country_code", "span_start", 
                "span_finish", "span_text", "pcl_category", "number_of_annotators"]
COLS_TO_ADD = ["span_start", "span_finish", "span_text", "pcl_category", "number_of_annotators"]
SEPARATOR = "\t"    # the separator used (and to use)
SKIP_ROWS = 4       # the first SKIP_ROWS represent a description of the data

# Label mappings
LAB2ID_T1 = {
    0: 0, 1: 0,         # negative label
    2: 1, 3: 1, 4: 1    # positive label
}
LAB2ID_T2 = {
    "Unbalanced_power_relations": 0,
    "Shallow_solution": 1,
    "Presupposition": 2,
    "Authority_voice": 3,
    "Metaphors": 4,
    "Compassion": 5,
    "The_poorer_the_merrier": 6
}


def merge_tasks_data():
    """A method that creates a data dictionary which stores information for both subtasks."""

    def attach_task2_data_to_dict(task2_data, data_dict):
        """A function that attaches task2 data to a dictionary already storing task1 data."""

        # Attach task2 information to the task1 dictionary
        for index, row in task2_data.iterrows():
            if row["par_id"] in data_dict.keys():
                for col in COLS_TO_ADD:
                    if col not in data_dict[row["par_id"]].keys():
                        data_dict[row["par_id"]][col] = [row[col]]
                    else:
                        data_dict[row["par_id"]][col].append(row[col])
            else:
                sys.exit(f"{row['par_id']} is not on the data_dict dictionary. Exit.")

        # Create empty attributes if there is no task2 information for them
        for id_, attrs in data_dict.items():
            for col in COLS_TO_ADD:
                if col not in attrs:
                    data_dict[id_][col] = []

        return data_dict
    
    # Read the raw task1 data and store it as dataframe
    df_task1 = pd.read_csv(DATA_FILEPATH_TASK1, 
        names=HEADER_TASK1[1:], skiprows=4, index_col=0, sep=SEPARATOR)

    # Caveat: entry 8640 has "nan" as text (replace it with an empty line)
    df_task1["text"] = df_task1["text"].fillna("")

    # Convert the cleaned dataframe to a dictionary format
    data_dict = df_task1.to_dict(orient="index")

    # Read the raw task2 data and store it as dataframe
    df_task2 = pd.read_csv(DATA_FILEPATH_TASK2, 
        names=HEADER_TASK2, skiprows=4, sep=SEPARATOR)

    # Attach task2 data to the previously created data dictionary
    data_dict = attach_task2_data_to_dict(df_task2, data_dict)

    return data_dict


def collect_split_ids(input_filepath):
    """A function that returns the IDs for the practice splits provided by organizers."""
    ids = list()
    is_header = True

    with open(input_filepath, "r") as f:
        for line in f:
            if is_header:
                is_header = False
            else:
                ids.append(line.split(",", 1)[0])

    return ids


def create_paragraph_data_view(data_dict, output_file_prefix):
    """A function that creates the paragraph data view files in the MaChAmp format
       according to the original train/dev splits provided by organizers. Test data
       will be converted to the MaChAmp format too."""

    # Collect instance IDs for creating the standard data split
    train_ids = collect_split_ids(PRACTICE_TRAIN_FILEPATH)
    dev_ids = collect_split_ids(PRACTICE_DEV_FILEPATH)

    # Create files for the standard data split
    train_file = open(os.path.join(DATA_FOLDER, output_file_prefix + ".train"), "w")
    dev_file = open(os.path.join(DATA_FOLDER, output_file_prefix + ".dev"), "w")

    # Dummy placeholder for removed/unknown fields
    blank = "0"

    # Iterate over the splits to create files in the MaChAmp format
    for (split_ids, output_file) in [(train_ids, train_file), (dev_ids, dev_file)]:
        for id_ in split_ids:
            # Get the fields that do not need further manipulation
            text = data_dict[int(id_)]["text"]
            keyword = data_dict[int(id_)]["keyword"]
            country = data_dict[int(id_)]["country_code"]
            raw_label_t1 = data_dict[int(id_)]["label"]
            raw_labels_t2 = data_dict[int(id_)]["pcl_category"]
            raw_spans_t2 = data_dict[int(id_)]["span_text"]
            raw_confidence_t2 = data_dict[int(id_)]["number_of_annotators"]

            # Convert the raw (uncertainty) label for subtask 1 into a binary format
            # according to the way PCL identification is formally evaluated
            label_t1 = LAB2ID_T1[raw_label_t1]

            # Get binary labels for subtask2 (either 0 or 1 for each PCL label)
            labels_t2 = {key: 0 for key in list(LAB2ID_T2.keys())}
            assert (len(raw_labels_t2) == len(raw_spans_t2) == len(raw_confidence_t2))
            for i in range(len(raw_labels_t2)):
                curr_label = raw_labels_t2[i]
                labels_t2[curr_label] = 1

            # Construct the line for the instance to be printed to a file
            line = SEPARATOR.join([
                str(id_), 
                text, 
                keyword, 
                country, 
                str(raw_label_t1), 
                blank, 
                str(label_t1), 
                str(labels_t2["Unbalanced_power_relations"]), 
                str(labels_t2["Shallow_solution"]), 
                str(labels_t2["Presupposition"]), 
                str(labels_t2["Authority_voice"]), 
                str(labels_t2["Metaphors"]), 
                str(labels_t2["Compassion"]), 
                str(labels_t2["The_poorer_the_merrier"]),
                ""
            ])

            # Actual print of the instance to the output file
            output_file.write(line + "\n")

        output_file.close()

    # Convert test data to the MaChAmp format
    test_file = open(os.path.join(DATA_FOLDER, output_file_prefix + ".test"), "w")
    with open(TEST_FILEPATH, "r") as f:
        for line in f:
            id_, id2_, keyword, country, text = line.rstrip().split("\t")
            output_line = "\t".join([id_, text, keyword, country, blank, blank, blank,
                blank, blank, blank, blank, blank, blank, blank, blank])
            test_file.write(output_line + "\n")
    test_file.close()


def create_span_data_view(output_file_prefix):
    """A function that creates the span data view files in the MaChAmp format
       according to the original train/dev splits provided by organizers."""

    # Collect instance IDs for creating the standard data split
    train_ids = collect_split_ids(PRACTICE_TRAIN_FILEPATH)
    dev_ids = collect_split_ids(PRACTICE_DEV_FILEPATH)

    # Create files for the standard data split
    train_file = open(os.path.join(DATA_FOLDER, output_file_prefix + ".train"), "w")
    dev_file = open(os.path.join(DATA_FOLDER, output_file_prefix + ".dev"), "w")

    # Read the raw task2 data and store it as dataframe
    df_task2 = pd.read_csv(DATA_FILEPATH_TASK2, 
        names=HEADER_TASK2, skiprows=4, sep=SEPARATOR)

    # Convert the dataframe to a dictionary format
    data_dict = df_task2.to_dict(orient="index")

    # Iterate over the data dictionary and get the fields
    for id_, instance in data_dict.items():
        par_id = str(instance["par_id"])
        art_id = instance["art_id"]
        text = instance["text"]
        keyword = instance["keyword"]
        country = instance["country_code"]
        span_start = str(instance["span_start"])
        span_end = str(instance["span_finish"])
        span_text = instance["span_text"]
        pcl_category = instance["pcl_category"]
        no_annotators = str(instance["number_of_annotators"])

        # Construct the line for the instance to be printed to a file
        line = SEPARATOR.join([
            par_id, art_id, text, keyword, country, span_start, span_end,
            span_text, pcl_category, no_annotators
        ])

        # Actual print of the instance to the output file
        if par_id in train_ids:
            train_file.write(line + "\n")
        elif par_id in dev_ids:
            dev_file.write(line + "\n")
        else:
            sys.exit(f"{par_id} is not on the paragraph ID lists. Exit.")

    train_file.close()
    dev_file.close()


if __name__ == "__main__":
    # Create a dictionary storing information for both subtasks
    data_dict = merge_tasks_data()

    # Create paragraph data view files in the MaChAmp format (original train/dev splits)
    create_paragraph_data_view(data_dict, output_file_prefix="paragraphs")

    # Create span data view files in the MaChAmp format (consistently to the original split)
    create_span_data_view(output_file_prefix="spans")
