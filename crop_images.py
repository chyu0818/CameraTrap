import os
import numpy as np
import json
from PIL import Image
from torchvision import transforms as T

def main():
    bbox_fn = '../efs/iwildcam2020_megadetector_results.json'
    im_fp = '../efs/train'
    im_fp_test = '../efs/test'
    im_crop_fp = '../efs/train_crop'
    im_crop_fp_test = '../efs/test_crop'
    transform = T.Resize(256)

    # Read in json file.
    with open(bbox_fn) as f1:
        detections = json.load(f1)['images']
    print(len(detections), 'expected number of images')
    total_crop = sum([len(im_info['detections']) for im_info in detections])
    print(total_crop, 'expected number of cropped images')

    counter = 0
    counter_crop = 0
    for im_info in detections:
        id = im_info['id']
        if len(im_info['detections']) > 0:
            # Read in image.
            try:
                im = Image.open(os.path.join(im_fp,'{}.jpg'.format(id)))
                (n_rows, n_cols, n_channels) = np.shape(im)
                for i in range(len(im_info['detections'])):
                    d = im_info['detections'][i]
                    [x, y, width, height] = d['bbox']
                    bbox = (int(x*n_cols), int(y*n_rows), int((x+width)*n_cols), int(n_rows*(y+height)))
                    # Crop image with PIL and resize so that smallest side is 256.
                    im_crop = transform(im.crop(bbox))
                    im_crop_fn = '{}_{}_{}.jpg'.format(id, i, '-'.join([str(dd) for dd in d['bbox']]))
                    im_crop.save(os.path.join(im_crop_fp,im_crop_fn))
                    counter_crop += 1
                counter += 1
            except IOError:
                im = Image.open(os.path.join(im_fp_test,'{}.jpg'.format(id)))
                (n_rows, n_cols, n_channels) = np.shape(im)
                for i in range(len(im_info['detections'])):
                    d = im_info['detections'][i]
                    [x, y, width, height] = d['bbox']
                    bbox = (int(x*n_cols), int(y*n_rows), int((x+width)*n_cols), int(n_rows*(y+height)))
                    # Crop image with PIL and resize so that smallest side is 256.
                    im_crop = transform(im.crop(bbox))
                    im_crop_fn = '{}_{}_{}.jpg'.format(id, i, '-'.join([str(dd) for dd in d['bbox']]))
                    im_crop.save(os.path.join(im_crop_fp_test,im_crop_fn))
                    counter_crop += 1
                counter += 1
                continue
    print(counter, 'number of images')
    print(counter_crop, 'number of cropped images')
    assert(counter == len(detections))
    assert(counter_crop == total_crop)



if __name__ == '__main__':
    main()
