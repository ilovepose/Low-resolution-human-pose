# import pycocotools.coco as coco
# import os
# import cv2
# import numpy as np
import _init_paths
# from datasets.dataset_factory import get_dataset
# from opts import opts, opts2
# import matplotlib.pyplot as plt
# import pylab


def add(d, keys, values):
    for i, key in enumerate(keys):
        d[key] = values[i]


def pickle_save(d):
    import pickle
    with open('debug.pickle', mode='ab+') as f:
        pickle.dump(d, f)


def pickle_read(name):
    import pickle
    with open(name, mode='rb') as f:
        try:
            d = {}
            while True:
                c = pickle.load(f)
                d.update(c)
        except EOFError:
            pass
    return d


def test(opt, ):
    current_dir = os.path.dirname(__file__)
    data_dir = os.path.join(current_dir, '../data/coco/annotations/person_keypoints_train2017.json')
    cocoins = coco.COCO('data/coco/annotations/person_keypoints_{}2017.json'.format('train'))

    image_ids = cocoins.getImgIds()
    index = 1

    file_name = cocoins.loadImgs(ids=[image_ids[index]])[0]['file_name']
    img_path = os.path.join('./data/coco/train2017', file_name)
    img = cv2.imread(img_path)

    idxs = cocoins.getAnnIds(imgIds=[image_ids[index]])
    anns = cocoins.loadAnns(ids=idxs)

    print(anns[0].keys())
    print(img.shape)
    ann = anns[0]
    bbox = ann['bbox']
    bbox = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]
    pts = np.array(ann['keypoints'], np.float32).reshape(-1, 3)
    mask = pts[:, 2] == 0
    mask = 1-mask
    min_x, min_y, _ = pts[mask].min(axis=0)
    max_x, max_y, _ = pts[mask].max(axis=0)
    print("bbox: ",bbox)
    print("joints: ", pts)
    print("left corner: ", min_x, min_y, ", right corner: ", max_x, max_y)


    Dataset = get_dataset('coco_hp', 'multi_pose')
    opt = opts2().update_dataset_info_and_set_heads(opt, Dataset)

    dataset = Dataset(opt, 'train')

    ret = dataset[1]

    print(ret['hm'].argmax())
    print(ret['hm_up'].argmax())
    print(ret['hm_bot'].argmax())

    plt.subplot(2, 2, 1)
    plt.imshow(ret['input'][0]*255)

    plt.subplot(2, 2, 2)
    plt.imshow(ret['hm'][0]*255)

    plt.subplot(2, 2, 3)
    plt.imshow(ret['hm_up'][0]*255)

    plt.subplot(2, 2, 4)
    plt.imshow(ret['hm_bot'][0]*255)

    plt.show()







if __name__ == '__main__':
    opt = opts2().parse()
    opts2.print()
    test(opt)
