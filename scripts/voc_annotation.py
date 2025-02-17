import os
import argparse
import xml.etree.ElementTree as ET

def convert_voc_annotation(data_path, anno_path, use_difficult_bbox=True):

    classes = ['rebar']
    image_inds = []
    img_inds_file = os.path.join(data_path, 'Annotations')
    for file_name in os.listdir(img_inds_file):
        image_inds.append(os.path.splitext(file_name)[0])

    with open(anno_path, 'a') as f:
        for image_ind in image_inds:
            image_path = os.path.join(data_path, 'JPEGImages', image_ind + '.jpg')
            annotation = image_path
            label_path = os.path.join(data_path, 'Annotations', image_ind + '.xml')
            root = ET.parse(label_path).getroot()
            objects = root.findall('object')
            for obj in objects:
                difficult = obj.find('difficult').text.strip()
                if (not use_difficult_bbox) and(int(difficult) == 1):
                    continue
                bbox = obj.find('bndbox')
                class_ind = classes.index(obj.find('name').text.lower().strip())
                xmin = bbox.find('xmin').text.strip()
                xmax = bbox.find('xmax').text.strip()
                ymin = bbox.find('ymin').text.strip()
                ymax = bbox.find('ymax').text.strip()
                annotation += ' ' + ','.join([xmin, ymin, xmax, ymax, str(class_ind)])
            print(annotation)
            f.write(annotation + "\n")
    return len(image_inds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="/home/yang/test/VOC/")
    parser.add_argument("--train_annotation", default="./data/dataset/voc_train.txt")
    parser.add_argument("--test_annotation",  default="./data/dataset/voc_test.txt")
    flags = parser.parse_args()

    if os.path.exists(flags.train_annotation):os.remove(flags.train_annotation)
    if os.path.exists(flags.test_annotation):os.remove(flags.test_annotation)

    num1 = convert_voc_annotation(os.path.join(flags.data_path, 'train_data_VOC'), flags.train_annotation, False)
    num3 = convert_voc_annotation(os.path.join(flags.data_path, 'test_VOC'),  flags.test_annotation, False)
    print('=> The number of image for train is: %d\tThe number of image for test is:%d' % (num1, num3))


