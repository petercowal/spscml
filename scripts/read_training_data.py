import json




with open("training_data.txt", "r") as file:
    for line in file:
        data = json.loads(line)
        print(data)
