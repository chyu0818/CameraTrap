import json
import numpy as np
import matplotlib.pyplot as plt

def plot_animals(categories):
    categories_sort = sorted(categories, key=lambda k: k['count'], reverse=True)
    names = []
    ids = []
    counts = []
    for i in range(len(categories)):
        names.append(categories_sort[i]['name'])
        ids.append(str(categories_sort[i]['id']))
        counts.append(categories_sort[i]['count'])

    plt.bar(ids[150:],counts[150:])
    plt.title('Bottom Categories')
    plt.xlabel('Ids')
    plt.ylabel('Number of Images')
    plt.show()

    plt.hist(counts[150:], bins=100)
    plt.xlabel('Counts')
    plt.title('Number of Species With Count (Bottom)')
    plt.show()
    return

def plot_locations(data):
    pass

def plot_time(data):
    pass

def main():
    with open('iwildcam2020_train_annotations.json') as f:
        data = json.load(f)
    print(data['categories'])
    plot_animals(data['categories'])

if __name__ == '__main__':
    main()
