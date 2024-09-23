import os.path as osp
import numpy as np
from torch.utils import data
from PIL import Image
import scipy.misc as m

class ACDCDataset(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128)):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.mean = mean
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        
        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [
            7,
            8,
            11,
            12,
            13,
            17,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            31,
            32,
            33,
        ]
        self.class_names = [
            "unlabelled",
            "road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic_light",
            "traffic_sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
        ]
        
        self.ignore_index = 255
        self.class_map = dict(zip(self.valid_classes, range(19)))
        for name in self.img_ids:
            img_file = osp.join(self.root, "./ACDC/%s" % (name))
            label_file = osp.join(self.root, "./ACDC/%s" % ("gt/"+name[9:][:-12]+"gt_labelTrainIds.png"))
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })
            
            
    def __len__(self):
        return len(self.files)
    
    
    def __getitem__(self, index):
        datafiles = self.files[index]
      
        image = Image.open(datafiles["img"]).convert('RGB')
        name = datafiles["name"]
        
        # resize
        w, h = image.size

        image = image.resize(self.crop_size, Image.BICUBIC)
        image = np.asarray(image, np.float32)


        size = image.shape
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        image = image.transpose((2, 0, 1))

        return image.copy(), np.array(size), name