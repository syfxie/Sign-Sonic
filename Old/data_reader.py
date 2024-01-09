# Delete later

import json

path = 'WLASL_v0.3.json'

with open(path) as file:
    data = json.load(file)

train_count, val_count, test_count = 0, 0, 0

for ent in data:
    gloss = ent['gloss']

    for inst in ent['instances']:
        split = inst['split']

        if split == 'train':
            train_count += 1
        elif split == 'val':
            val_count += 1
        elif split == 'test':
            test_count += 1
        else:
            raise ValueError("Invalid split.")

print('total glosses: {}'.format(len(data)))
print('total samples: {}'.format(train_count + val_count + test_count))
