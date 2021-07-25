import json

filename = "/home/beecadox/Thesis/Dataset/ShapeNetCore.v2/taxonomy.json"
f = open(filename)
taxonomy = json.load(f)

with open('/home/beecadox/Thesis/Dataset/ShapeNetCore.v2/objects.txt') as f:
    objects = f.read().splitlines()

dataset = {}
for tax in taxonomy:
    if tax['synsetId'] in objects:
        dataset[tax['synsetId']] = tax['name']

# Write-Overwrites
file1 = open("objects.txt", "w") # write mode
for datum in dataset:
    file1.write(datum + " " + dataset[datum] + "\n")
file1.close()