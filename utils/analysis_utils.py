import json

success_count = 0
total_count = 0

results_file_path = "/result.jsonl"

try:
    total = 0
    success_count = 0
    with open(results_file_path, "r") as f:
        for line in f:
            data = line.split(" ")
            if len(data) == 2 and (data[0] == "True" or data[0] == "False"):
                total_count += 1
                if data[1] == "True\n":
                    success_count += 1

    if total_count > 0:
        success_rate = (success_count / total_count) * 100
        print(f"Total runs: {total_count}")
        print(f"Successful runs: {success_count}")
        print(f"Success Rate: {success_rate:.2f}%")
    else:
        print("No results found in the file.")
except FileNotFoundError:
    print(
        f"Error: The file '{results_file_path}' was not found. Please ensure it exists after running the model."
    )
except Exception as e:
    print(f"An error occurred while processing the file: {e}")
