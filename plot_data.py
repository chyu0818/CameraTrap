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

    plt.bar(ids[:8],counts[:8])
    print(names[:8])
    plt.title('Top 8 Categories')
    plt.xlabel('Ids')
    plt.ylabel('Number of Images')
    plt.show()

    plt.hist(counts[150:], bins=100)
    plt.xlabel('Counts')
    plt.title('Number of Species With Count (Bottom)')
    plt.show()
    return

def plot_locations(data, category_data, annotations):
    # There are 552 different locations
    locations = list(range(552))
    count = [0 for i in range(552)]
    for image in data:
        count[image['location']] += 1
    # plt.bar(locations_sort[-20:], counts_sorted[-20:])

    # Plot general counts of images per location
    plt.bar(locations, count)
    plt.title('Number of Images Per Location')
    plt.xlabel('Location ID')
    plt.ylabel('Number of Images')
    plt.show()

    # Ascending order of locations with most images
    locations_sort = np.argsort(count)
    locs = locations_sort[-20:]
    valid_images = {}
    count_by_loc = {}
    unique_categories = set([])

    # Collect species count data on twenty most populated regions
    for image in data:
        if image['location'] in locs:
            valid_images[image['id']] = image['location']
            count_by_loc[image['location']] = {}
    for item in annotations:
        if item['image_id'] in valid_images:
            cat = item['category_id']
            if cat not in unique_categories:
                unique_categories.add(cat)
            if cat in count_by_loc[valid_images[item['image_id']]]:
                count_by_loc[valid_images[item['image_id']]][cat] += 1
            else:
                count_by_loc[valid_images[item['image_id']]][cat] = 1

    # For each category, get it's name
    cat_names = {}
    for item in category_data:
        if item['id'] in unique_categories:
            cat_names[item['id']] = item['name']

    names = []
    indices = list(range(20))
    base = np.array([0 for _ in range(20)])
    for j, item in enumerate(unique_categories):
        # Get counts of this category for each location and plot
        names.append(cat_names[item])
        counts = [0 for _ in range(20)]
        for i, loc in enumerate(locs):
            if item in count_by_loc[loc]:
                counts[i] = count_by_loc[loc][item]
        if j == 0:
            plt.bar(indices, counts)
        else:
            plt.bar(indices, counts, bottom=base)
        base = base + np.array(counts)
    plt.xticks(indices, list(map(lambda x: str(x), locs)), rotation=60)
    plt.xlabel('Location ID')
    plt.ylabel('Count Divided by Species')
    plt.title('Species Breakdown of Locations with Most Images')
    plt.legend(names, prop={'size': 6}, loc='upper left')
    plt.show()
    return


def create_im_dict(annotations, detections):
    # Create dictionary with the key as image id and the values as
    # the bounding boxes and category id.
    # detections is a list of dictionaries {category 1/2, bbox, confidence}
    im_dict = {a['image_id']:{'category_id':a['category_id']} for a in annotations}
    for im in im_dict:
        im_dict[im]['detections'] = []

    # Iterate over bounding boxes info.
    for detect in detections:
        if im_dict.get(detect['id']) != None:
            # if im_dict[detect['id']].get('detections') != None:
            #     print('ERROR: More than one detection')
            im_dict[detect['id']]['detections'] = detect['detections']
    return im_dict

def plot_detect_conf(detections, title):
    print(title)
    print(len(detections))
    print(len(np.argwhere(np.asarray(detections) >= 0.4)), '\n')
    # plt.hist(detections, bins=10)
    # plt.xlabel('Confidence')
    # plt.title('Number of Detections with Confidence Score ({})'.format(title))
    # plt.show()

def extract_detections(im_dict, file_lst):
    detections = []
    for im in file_lst:
        for d in im_dict[im]['detections']:
            detections.append(d['conf'])
    return detections

def plot_detect_conf_all(annotations, detections):
    # Both train and test
    im_dict = create_im_dict(annotations['annotations'], detections)
    detections_all = [d['conf'] for im in im_dict for d in im_dict[im]['detections']]
    detections_all.sort(reverse=True)
    plot_detect_conf(detections_all, 'all')

    # Our train
    file_lst = np.load('X_train.npz')['arr_0']
    detections = extract_detections(im_dict, file_lst)
    plot_detect_conf(detections, 'train')

    # Our cis val
    file_lst = np.load('X_val_cis.npz')['arr_0']
    detections = extract_detections(im_dict, file_lst)
    plot_detect_conf(detections, 'val cis')

    # Our trans val
    file_lst = np.load('X_val_trans.npz')['arr_0']
    detections = extract_detections(im_dict, file_lst)
    plot_detect_conf(detections, 'val trans')

    # Our cis test
    file_lst = np.load('X_test_cis.npz')['arr_0']
    detections = extract_detections(im_dict, file_lst)
    plot_detect_conf(detections, 'test cis')

    # Our trans test
    file_lst = np.load('X_test_trans.npz')['arr_0']
    detections = extract_detections(im_dict, file_lst)
    plot_detect_conf(detections, 'test trans')


def main():
    with open('iwildcam2020_train_annotations.json') as f:
        annotations = json.load(f)
    plot_animals(annotations['categories'])
    # plot_locations(annotations['images'], annotations['categories'], annotations['annotations'])

    # with open('iwildcam2020_megadetector_results.json') as f:
    #     detections = json.load(f)['images']

    # plot_detect_conf_all(annotations, detections)

if __name__ == '__main__':
    main()
