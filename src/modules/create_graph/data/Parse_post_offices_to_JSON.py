import json
import csv

csv_path = "./PostalOffices_PoC_2.csv"
json_path = "./PostalOffices_PoC.json"

data = {}
data2 = {}
with open(csv_path) as csvFile:
    csvReader = csv.DictReader(csvFile)
    for rows in csvReader:
        #print(rows["address"])
        id = rows["uuid"]
        data[id] = rows
        #data2["posts"].append(rows)

with open(json_path, "w") as jsonFile:
    jsonFile.write(json.dumps(data, indent=4))


