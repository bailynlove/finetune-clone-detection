import subprocess
import xml.etree.ElementTree as ET
import glob
import shutil
import pandas as pd
import os
from pathlib import Path
import ConfigParser


def delete_folder(folder: str):
    """delete folder

    Args:
        folder (str): which folder to delete
    """

    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file_path, e))


def find_input_function_lines(file_loc: str) -> list:
    """find input code lines

    Args:
        file_loc (str): input file location

    Returns:
        list: start and end line of input code
    """

    file_line = open(file_loc, "r")
    input_found = False
    input_func_lines = []
    for num, line in enumerate(file_line, 0):
        if "input".casefold() in line.casefold():
            input_func_lines.append(num + 1)
            input_found = True
        if input_found and line.startswith("}"):
            input_func_lines.append(num + 1)
            break
    return input_func_lines


def find_py_input_function_lines(file_loc: str) -> list:
    """find all py functions from raw GPT output

    Args:
        file_loc (str): raw GPT output file

    Returns:
        list: list of start and end lines of each code
    """

    file_line = open(file_loc, "r")
    input_found = False
    input_func_lines = []
    for num, line in enumerate(file_line, 0):
        if "#input".casefold() in line.casefold():
            input_func_lines.append(num + 1)
            input_found = True
        if input_found and "#====================" in line.casefold():
            input_func_lines.append(num - 1)
            break
    return input_func_lines


def find_type_3_lines(file_loc: str) -> list:
    """find all type-3 functions from raw GPT output

    Args:
        file_loc (str): raw GPT output file

    Returns:
        list: list of start and end lines of each code
    """

    file_line = open(file_loc, "r")
    type3_found = False
    type3_lines = []
    search_term = ["Type 3", "Type-3", "Type3"]
    type3_start = -1
    for num, line in enumerate(file_line, 0):
        if type3_found or any(
            search.casefold() in line.casefold() for search in search_term
        ):
            if not type3_found:
                type3_start = num + 1
            type3_found = True
            if type3_found and line.startswith("}"):
                type3_lines.append([type3_start, num + 1])
                type3_found = False
                type3_start = -1
    return type3_lines


def find_py_def_lines(file_name: str, lookup: str) -> list[int]:
    """find all py functions from raw GPT output

    Args:
        file_loc (str): raw GPT output file

    Returns:
        list: list of start and end lines of each code
    """

    function_lines = []
    last_line_num = -1

    with open(file_name) as myFile:
        for num, line in enumerate(myFile, 0):
            if lookup in line:
                function_lines.append(num)
            last_line_num = num

    if len(function_lines) <= 1:
        return []
    new_func_lines = function_lines[1:]
    all_func = []
    for i in range(0, len(new_func_lines) - 1):
        all_func.append([new_func_lines[i], new_func_lines[i + 1] - 2])
    all_func.append([new_func_lines[-1], last_line_num])
    return all_func


def find_type_4_lines(file_loc: str) -> list:
    """find all type-4 functions from raw GPT output

    Args:
        file_loc (str): raw GPT output file

    Returns:
        list: list of start and end lines of each code
    """

    file_line = open(file_loc, "r")
    type4_found = False
    type4_lines = []
    search_term = ["Type 4", "Type-4", "Type4"]
    type_4_start = -1
    for num, line in enumerate(file_line, 0):
        if type4_found or any(
            search.casefold() in line.casefold() for search in search_term
        ):
            if not type4_found:
                type_4_start = num + 1
            type4_found = True
            if type4_found and line.startswith("}"):
                type4_lines.append([type_4_start, num + 1])
                type4_found = False
                type_4_start = -1
    return type4_lines


def create_ignore_list(file: str) -> list[str]:
    """reading all clone folders

    Args:
        file (str): location of file

    Returns:
        list[str]: list of all clone folders
    """

    text_file = open(file, "r", encoding="utf-8", errors="ignore")
    lines = text_file.readlines()
    ignore_list = []
    for line in lines:
        if line.replace(" ", "") != "" or line.replace(" ", "") != "\n":
            ignore_list.append(line.replace("\n", ""))
    return ignore_list


def main():
    config = ConfigParser.ConfigParser()
    config.readfp(open(r"type34_config.txt"))
    target_lang = config.get("config", "target_lang")
    input_dir = config.get("config", "input_dir")
    nicad_output = config.get("config", "nicad_output")
    nicad_folder = config.get("config", "nicad_folder")
    dest_type3 = config.get("config", "dest_type3")
    dest_type4 = config.get("config", "dest_type3")
    semantic_output = config.get("config", "semantic_output")
    raw_output = config.get("config", "raw_output")
    ignore_file_list = config.get("config", "ignore_file_list")
    type3_cap = config.get("config", "type3_cap")
    type4_cap = config.get("config", "type4_cap")
    nicad_loc = config.get("config", "nicad_loc")

    all_files = sorted(glob.glob(input_dir + target_lang + "/*." + target_lang))

    input_file_name = (
        nicad_output + target_lang + "_" + nicad_folder + "/input." + target_lang
    )

    output = nicad_output + target_lang + "_" + nicad_folder + "/output"

    Path(dest_type3).mkdir(parents=True, exist_ok=True)
    Path(dest_type4).mkdir(parents=True, exist_ok=True)
    Path(nicad_output).mkdir(parents=True, exist_ok=True)

    semantic_data = pd.DataFrame(columns=["file", "similarity"])
    raw_results = pd.DataFrame(columns=["input", "output", "similarity"])

    ignored_file = create_ignore_list(ignore_file_list)
    total_processed_file = 0

    for file in all_files:
        if file.split("/")[-1] not in ignored_file:
            continue

        if target_lang != "py":
            input_lines = find_input_function_lines(file)
            type3_lines = find_type_3_lines(file)

        else:
            input_lines = find_py_input_function_lines(file)
            type3_lines = find_py_def_lines(file, "def")

        type4_lines = find_type_4_lines(file)

        if len(type3_lines) <= 0 and len(type4_lines) <= 0:
            print("[ERROR Happened] : No type 3 or 4")
            continue

        Path(nicad_output + "/" + target_lang + "_" + nicad_folder).mkdir(
            parents=True, exist_ok=True
        )
        text_file = open(file, "r")
        lines = text_file.readlines()
        input_file = lines[input_lines[0] : input_lines[1]]
        input_str = ""
        with open(input_file_name, "w") as f:
            for i in input_file:
                f.write(i)
                input_str += i
        file_create_counter = 0
        for index in range(0, len(type3_lines)):
            output_file_name = output + "_" + str(file_create_counter) + ".c"
            output_file = lines[type3_lines[index][0] : type3_lines[index][1]]
            with open(output_file_name, "w") as f:
                for i in output_file:
                    f.write(i)
            file_create_counter += 1

        for index in range(0, len(type4_lines)):
            output_file_name = output + "_" + str(file_create_counter) + ".c"
            output_file = lines[type4_lines[index][0] : type4_lines[index][1]]
            with open(output_file_name, "w") as f:
                for i in output_file:
                    f.write(i)
            file_create_counter += 1

        command = (
            "cd "
            + nicad_loc
            + "./nicad6 functions "
            + target_lang
            + " "
            + os.path.abspath(os.getcwd())
            + "/"
            + nicad_output
            + "/"
            + target_lang
            + "_test default"
        )

        ret = subprocess.run(command, capture_output=True, shell=True)
        tree = None
        try:
            tree = ET.parse(
                nicad_output
                + "/"
                + target_lang
                + "_test_functions-blind-clones/"
                + target_lang
                + "_test_functions-blind-clones-0.99.xml"
            )
        except Exception as e:
            delete_folder(nicad_output + "/")
            print("[ERROR Happened] : ", e)
            continue
        root = tree.getroot()
        pair_counter = 0
        for child in root:
            similarity = []
            if child.tag == "clone":
                found_similarity = int(child.attrib["similarity"])
                if found_similarity <= type3_cap:
                    file_list = []
                    input_file_found = False
                    input_file_index = -1
                    for inner_child in child:
                        file_name = inner_child.attrib["file"]
                        if file_name.split("/")[-1] == "input.c":
                            input_file_found = True
                        file_list.append(file_name)
                    if not input_file_found:
                        file_list = []
                        continue
                    all_code = ""
                    input_index = -1
                    for file_index in range(0, len(file_list)):
                        if file_list[file_index].split("/")[-1] == "input.c":
                            input_index = file_index
                            break

                    file_list.insert(0, file_list.pop(input_index))

                    for f in file_list:
                        file_line = open(f, "r")
                        all_lines = file_line.readlines()
                        for l in all_lines:
                            all_code += l
                        all_code += "\n\n"
                    file_name = (
                        file.split("/")[-1].split(".")[0]
                        + "_"
                        + str(pair_counter)
                        + ".c"
                    )
                    location = ""
                    if found_similarity > type4_cap:
                        output_file_name = dest_type3 + "/" + file_name
                        pair_counter += 1
                        location = output_file_name
                        total_processed_file += 1
                        with open(output_file_name, "w") as f:
                            f.write(all_code)

                    if found_similarity <= type4_cap:
                        output_file_name = dest_type4 + "/" + file_name
                        pair_counter += 1
                        location = output_file_name
                        total_processed_file += 1
                        with open(output_file_name, "w") as f:
                            f.write(all_code)

                    semantic_data.loc[len(semantic_data)] = [location, found_similarity]
                    input_code = ""
                    output_code = ""
                    for f in file_list:
                        all_code = ""
                        file_line = open(f, "r")

                        all_lines = file_line.readlines()
                        for l in all_lines:
                            all_code += l
                        if f.split("/")[-1] == "input.c":
                            input_code = all_code
                        else:
                            output_code = all_code
                    raw_results.loc[len(raw_results)] = [
                        input_code,
                        output_code,
                        found_similarity,
                    ]
                else:
                    file_list = []
                    input_file_found = False
                    for inner_child in child:
                        file_name = inner_child.attrib["file"]
                        if file_name.split("/")[-1] == "input.c":
                            input_file_found = True
                        file_list.append(file_name)
                    if not input_file_found:
                        file_list = []
                        continue

                    input_code = ""
                    output_code = ""
                    for f in file_list:
                        all_code = ""
                        file_line = open(f, "r")

                        all_lines = file_line.readlines()
                        for l in all_lines:
                            all_code += l
                        if f.split("/")[-1] == "input.c":
                            input_code = all_code
                        else:
                            output_code = all_code
                    raw_results.loc[len(raw_results)] = [
                        input_code,
                        output_code,
                        found_similarity,
                    ]
        delete_folder(nicad_output + "/")

    semantic_data.to_csv(semantic_output, index=False)
    raw_results.to_csv(raw_output, index=False)
    print("total_processed_file: ", total_processed_file)
