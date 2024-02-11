import json

import os


set1 = json.load(open("detection/data/GBCU-Shared/annotations/train_new.json"))
set2 = json.load(open("detection/data/GBCU-Shared/annotations/test_new.json"))

# print(set1)

categories = set1['categories']
assert categories == set2['categories']


annotations1 = set1['annotations']
annotations2 = set2['annotations']

print(len(annotations1), len(annotations2))
print(len(annotations1)/(len(annotations1)+ len(annotations2)))
print(len(annotations2)/(len(annotations1)+ len(annotations2)))


annotations = annotations1+annotations2
print(len(annotations))


images1 = set1['images']
images2 = set2['images']

images = images1+images2
print(len(images))

print(images[0])