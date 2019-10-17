import cv2
import os
import numpy
from xml.dom.minidom import Document


def make_dir(root):
    if not os.path.exists(os.path.join(root, 'Annotations')):
        os.mkdir(os.path.join(root, 'Annotations'))
    if not os.path.exists(os.path.join(root, 'ImageSets')):
        os.makedirs(os.path.join(root, 'ImageSets/Main'))
    if not os.path.exists(os.path.join(root, 'JPEGImages')):
        os.mkdir(os.path.join(root, 'JPEGImages'))


def writexml(filename, saveimg, bboxes, xmlpath):
    doc = Document()

    annotation = doc.createElement('annotation')

    doc.appendChild(annotation)

    folder = doc.createElement('folder')

    folder_name = doc.createTextNode('widerface')
    folder.appendChild(folder_name)
    annotation.appendChild(folder)
    filenamenode = doc.createElement('filename')
    filename_name = doc.createTextNode(filename)
    filenamenode.appendChild(filename_name)
    annotation.appendChild(filenamenode)
    source = doc.createElement('source')
    annotation.appendChild(source)
    database = doc.createElement('database')
    database.appendChild(doc.createTextNode('wider face Database'))
    source.appendChild(database)
    annotation_s = doc.createElement('annotation')
    annotation_s.appendChild(doc.createTextNode('PASCAL VOC2007'))
    source.appendChild(annotation_s)
    image = doc.createElement('image')
    image.appendChild(doc.createTextNode('flickr'))
    source.appendChild(image)
    flickrid = doc.createElement('flickrid')
    flickrid.appendChild(doc.createTextNode('-1'))
    source.appendChild(flickrid)
    owner = doc.createElement('owner')
    annotation.appendChild(owner)
    flickrid_o = doc.createElement('flickrid')
    flickrid_o.appendChild(doc.createTextNode('muke'))
    owner.appendChild(flickrid_o)
    name_o = doc.createElement('name')
    name_o.appendChild(doc.createTextNode('muke'))
    owner.appendChild(name_o)

    size = doc.createElement('size')
    annotation.appendChild(size)

    width = doc.createElement('width')
    width.appendChild(doc.createTextNode(str(saveimg.shape[1])))
    height = doc.createElement('height')
    height.appendChild(doc.createTextNode(str(saveimg.shape[0])))
    depth = doc.createElement('depth')
    depth.appendChild(doc.createTextNode(str(saveimg.shape[2])))

    size.appendChild(width)

    size.appendChild(height)
    size.appendChild(depth)
    segmented = doc.createElement('segmented')
    segmented.appendChild(doc.createTextNode('0'))
    annotation.appendChild(segmented)
    for i in range(len(bboxes)):
        bbox = bboxes[i]
        objects = doc.createElement('object')
        annotation.appendChild(objects)
        object_name = doc.createElement('name')
        object_name.appendChild(doc.createTextNode('face'))
        objects.appendChild(object_name)
        pose = doc.createElement('pose')
        pose.appendChild(doc.createTextNode('Unspecified'))
        objects.appendChild(pose)
        truncated = doc.createElement('truncated')
        truncated.appendChild(doc.createTextNode('0'))
        objects.appendChild(truncated)
        difficult = doc.createElement('difficult')
        difficult.appendChild(doc.createTextNode('0'))
        objects.appendChild(difficult)
        bndbox = doc.createElement('bndbox')
        objects.appendChild(bndbox)
        xmin = doc.createElement('xmin')
        xmin.appendChild(doc.createTextNode(str(bbox[0])))
        bndbox.appendChild(xmin)
        ymin = doc.createElement('ymin')
        ymin.appendChild(doc.createTextNode(str(bbox[1])))
        bndbox.appendChild(ymin)
        xmax = doc.createElement('xmax')
        xmax.appendChild(doc.createTextNode(str(bbox[0] + bbox[2])))
        bndbox.appendChild(xmax)
        ymax = doc.createElement('ymax')
        ymax.appendChild(doc.createTextNode(str(bbox[1] + bbox[3])))
        bndbox.appendChild(ymax)
    f = open(xmlpath, "w")
    f.write(doc.toprettyxml(indent=''))
    f.close()


def to_voc(root):
    wire_face_dir = root
    wire_face_train_dir = os.path.join(wire_face_dir, 'train')
    wire_face_val_dir = os.path.join(wire_face_dir, 'valid')

    gt_train = os.path.join(wire_face_dir, 'split/wider_face_train_bbx_gt.txt')
    gt_val = os.path.join(wire_face_dir, 'split/wider_face_val_bbx_gt.txt')

    out_train = os.path.join(wire_face_dir, 'voc/ImageSets/Main/train.txt')
    out_val = os.path.join(wire_face_dir, 'voc/ImageSets/Main/val.txt')

    for img_root, gtfile, outlist in [(wire_face_train_dir, gt_train, out_train), (wire_face_val_dir, gt_val, out_val)]:
        with open(outlist, 'w') as writer:
            with open(gtfile, 'r') as gt:
                while True:
                    img_name = gt.readline()[:-1]
                    if img_name is None or img_name == '':
                        break

                    img_path = os.path.join(img_root, img_name)
                    im = cv2.imread(img_path)
                    if im is None:
                        continue

                    # cv2.imshow('im', im)
                    # cv2.waitKey(0)
                    sc = max(im.shape)
                    im_data_tmp = numpy.zeros([sc, sc, 3], dtype=numpy.uint8)
                    off_w = (sc - im.shape[1]) // 2
                    off_h = (sc - im.shape[0]) // 2

                    ##对图片进行周围填充，填充为正方形
                    im_data_tmp[off_h:im.shape[0] + off_h, off_w:im.shape[1] + off_w, ...] = im
                    im = im_data_tmp

                    # cv2.imshow('im', im)
                    # cv2.waitKey(0)

                    file_name = img_name.split('/')[-1]
                    save_path = "{}/voc/JPEGImages/{}".format(wire_face_dir, file_name)
                    cv2.imwrite(save_path, im)

                    writer.write(file_name.split(',')[0]+'\n')

                    bbox_nums = int(gt.readline()[:-1])
                    bboxes = []
                    for i in range(bbox_nums):
                        bbox = gt.readline()[:-1]
                        x, y, w, h = bbox.split(' ')[:4]
                        x, y, w, h = int(x)+off_w, int(y)+off_h, int(w), int(h)
                        bbox = (x, y, w, h)
                        bboxes.append(bbox)

                    xmlpath = "{}/voc/Annotations/{}.xml".format(wire_face_dir, file_name.split(".")[0])
                    writexml(file_name, im, bboxes, xmlpath)


if __name__ == '__main__':
    # to_voc('./')
    make_dir('d:/')
