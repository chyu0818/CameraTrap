import os
import numpy as np
import json
from PIL import Image, ImageDraw, ImageFont
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
                    if bbox[0] == bbox[2]:
                        bbox = (bbox[0], bbox[1], bbox[2]+1, bbox[3])
                    if bbox[1] == bbox[3]:
                        bbox = (bbox[0], bbox[1], bbox[2], bbox[3]+1)
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
                    if bbox[0] == bbox[2]:
                        bbox = (bbox[0], bbox[1], bbox[2]+1, bbox[3])
                    if bbox[1] == bbox[3]:
                        bbox = (bbox[0], bbox[1], bbox[2], bbox[3]+1)
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

def _draw_single_box(image, bbox, display_str, font, color='black', thickness=4):
    (xmin, ymin, xmax, ymax) = bbox
    draw = ImageDraw.Draw(image)
    (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    draw.line([(left, top), (left, bottom), (right, bottom),
             (right, top), (left, top)], width=thickness, fill=color)
    text_bottom = bottom
    # Reverse list and print from bottom to top.
    text_width, text_height = font.getsize(display_str)
    margin = np.ceil(0.05 * text_height)
    draw.rectangle(
      [(left, text_bottom - text_height - 2 * margin), (left + text_width,
                                                        text_bottom)],
      fill=color)
    draw.text(
      (left + margin, text_bottom - text_height - margin),
      display_str,
      fill='black',
      font=font)

    return image 



if __name__ == '__main__':
    bbox_fn = 'iwildcam2020_megadetector_results.json'
    # Read in json file.
    with open(bbox_fn) as f1:
        detections = json.load(f1)['images']
    with open('iwildcam2020_train_annotations.json') as f1:
        annotations0 = json.load(f1)
    categories = annotations0['categories']
    annotations = annotations0['annotations']
    print(categories)
    total_crop = sum([len(im_info['detections']) for im_info in detections])

    im_info = None
    for im_info0 in detections:
        if im_info0['id'] == '8678346c-21bc-11ea-a13a-137349068a90':
            im_info = im_info0
            break

    for ann in annotations:
        if ann['image_id'] == '8678346c-21bc-11ea-a13a-137349068a90':
            print(ann['category_id'])
            break


    # im = Image.open('../90717bfe-21bc-11ea-a13a-137349068a90.jpg')
    # draw = ImageDraw.Draw(im)
    # (n_rows, n_cols, n_channels) = np.shape(im)
    # for i in range(len(im_info['detections'])):
    #     d = im_info['detections'][i]
    #     [x, y, width, height] = d['bbox']
    #     bbox = (int(x*n_cols), int(y*n_rows), int((x+width)*n_cols), int(n_rows*(y+height)))
    #     draw.rectangle(bbox)
    # im.save('cropped.png')

    # main()
