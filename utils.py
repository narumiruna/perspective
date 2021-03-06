import json


def load_json(f):
    with open(f, 'r') as fp:
        return json.load(fp)

def save_json(obj, f):
    with open(f, 'w') as fp:
        json.dump(obj, fp, indent=2)
