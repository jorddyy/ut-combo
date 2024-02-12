import json
import pdb
import os, sys
import argparse

# argument for type of sample
parser = argparse.ArgumentParser()
parser.add_argument('--mode', '-m', 
    dest="mode",
    default='pi+ pi-',
    help="Mode to read from the input file")

def print_message(message: str):
    """
    Print a message in a fancy style
    """
    print(f"\n{'-'*60}\n{message}\n{'-'*60}\n")


def parse_input_file(input_file, mode):
    """
    Parse the inputs from the old inputs and return a ResultSet json file
    Tested just for uudMeasurements.list
    """

    # Information on the input file format
    """
    Lines with values
    0. Experiment = ResultSetLabel
    1. System name = name
    2. Sample size -> ignore
    3. Parameter name 
    4. Central value
    5. Error flag
      1: Seems to be just the statistical error
      5: Seems to be stat and syst separate
    6. Stat error
    7. Syst error
    7. or 8. Paper -> Description

    Lines with correlation
    0. Experiment = ResultSetLabel
    1. System name = name
    2. CORRELATION
    3. Parameter name 1
    4. Parameter name 2
    5. Correlation flag
      1:  Next is the off-diagonal statistical correlation
      3:  Next are the off-diagonal statistical and systematic correlations
    """

    results = {}

    with open(input_file, 'r') as file:

        # Make sure the script start reading until it gets to the right mode
        start_reading = False
        for line in file:

            # Find the right mode
            if f'### {mode} ###' in line:
                start_reading = True
            # Stop reading here
            elif '###' in line and start_reading:
                break
            # Skip comments
            elif line.startswith('#'):
                continue

            # Skip to next line if not in the right mode
            if not start_reading:
                continue

            parts = line.strip().split('\t')

            # Neither a line with values nor a line with correlation
            if len(parts) < 6:
                continue

            # Common fields
            experiment = parts[0].strip("'")
            mode = parts[1].strip("'")

            if len(parts) >= 8:
                corrLine = False
                desc = parts[-1].strip("'")
            else:
                corrLine = True

            # First time encountering the experiment
            if experiment not in results:
                results[experiment] = {
                    "ResultSetLabel": experiment,
                    "Description": [desc],
                    "Parameter": [],
                }
            
            # Normal line with parameter + values
            if not corrLine:
                parameter_name = parts[3]
                value = float(parts[4])

                errFlag = int(parts[5])

                if errFlag == 1:
                    error_stat = float(parts[6])
                    error_syst = None
                elif errFlag == 5:
                    error_stat = float(parts[6])
                    error_syst = float(parts[7])
                else:
                    raise ValueError(f"Unknown error flag: {errFlag}")

                parameter = {
                    "Name": parameter_name,
                    "Value": value,
                    "Error": error_stat
                }
                
                results[experiment]["Parameter"].append(parameter)
                
                if "SystematicErrors" not in results[experiment] and errFlag == 5:
                    results[experiment]["SystematicErrors"] = [{
                        "Name": "TotalSyst",
                        "Values": [error_syst]
                    }]
                elif errFlag == 5:
                    results[experiment]["SystematicErrors"][0]["Values"].append(error_syst)

            # Line with correlation
            else:
                # for now: 2x2 matrices only
                corrFlag = int(parts[5])

                # for now only flag 1 implemented
                if corrFlag == 1: # stat only
                    corr_value = float(parts[6])
                results[experiment]["StatisticalCorrelationMatrix"] = [
                    [1.0, corr_value],
                    [corr_value, 1.0]
                ]
    return list(results.values())

def write_to_json(data, output_file):
    json_data = {"ResultSet": data}
    print_message(f"Writing to {output_file}")
    with open(output_file, 'w') as file:
        json.dump(json_data, file, indent=4)

if __name__ == "__main__":
    # parse the arguments
    try:
        args = parser.parse_args()
        print_message("Input arguments")
        print(args)
    except:
        parser.print_help()
        sys.exit(0)

    mode = args.mode
    input_file = "inputs/old_inputs/uudMeasurements.list"
    # remove whitespace from mode
    output_file = f"ResultList_{mode.replace(' ', '')}.json"

    # Extract uudMeasurements string from the input file
    output_folder = f"inputs/{input_file.split('/')[-1].split('.')[0]}"

    # check if the folder exists and otherwise create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_path = f"{output_folder}/{output_file}"

    data = parse_input_file(input_file, mode)
    write_to_json(data, output_path)
