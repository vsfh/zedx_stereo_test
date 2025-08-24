import os
import cv2 as cv
import torch
from torch.utils.data import Dataset, DataLoader
cv.setNumThreads(0)
import random
import numpy as np
import torchvision.transforms as transforms
from tqdm import tqdm
from easydict import EasyDict as edict
import numpy as np
import albumentations as A
from albumentations import ImageOnlyTransform

TRAIN_LIST_ROOT = "/data/home/su0251/run/data/data_lists"
TEST_LIST_ROOT = "/data/home/su0251/run/data/data_lists"

config = edict()
config.train_list = [
    f"{TRAIN_LIST_ROOT}/zedx_train.list",
]
config.test_list = [
    f"{TEST_LIST_ROOT}/zedx_val.list",
]

config.num_classes = 2
config.num_epochs = 200
config.decay_epoch = [100]

# 1200 x 1920
config.H = 1200
config.W = 1920
config.focal_length = 738.193
config.baseline = 119.987
config.K = np.array([
    [738.193, 0.0, 966.136],
    [0.0, 737.878, 588.739],
    [0.0, 0.0, 1]
])

INPUT_H = 400
INPUT_W = 640
RAW_H = 1200
RAW_W = 1920




class SaltPepperNoise(ImageOnlyTransform):
    """Apply salt-and-pepper noise to the input image.

    Args:
        salt_prob ((float, float) or float): probability range of salt point for noise. If salt_prob is a single float, the range will be (0, salt_prob). Default: (0.01, 0.05).
        pepper_prob ((float, float) or float): probability range of pepper for noise. If pepper_prob is a single float, the range will be (0, pepper_prob). Default: (0.01, 0.05).
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(self, salt_prob=(0.01, 0.05), pepper_prob=(0.01, 0.05), always_apply=False, p=0.5):
        super(SaltPepperNoise, self).__init__(p=p, always_apply=always_apply)
        if isinstance(salt_prob, (tuple, list)):
            if salt_prob[0] < 0:
                raise ValueError("Lower salt_prob should be non negative.")
            if salt_prob[1] < 0:
                raise ValueError("Upper salt_prob should be non negative.")
            self.salt_prob = salt_prob
        elif isinstance(salt_prob, (int, float)):
            if salt_prob < 0:
                raise ValueError("salt_prob should be non negative.")

            self.salt_prob = (0, salt_prob)
        else:
            raise TypeError(
                "Expected salt_prob type to be one of (int, float, tuple, list), got {}".format(type(salt_prob))
            )
            
        if isinstance(pepper_prob, (tuple, list)):
            if pepper_prob[0] < 0:
                raise ValueError("Lower pepper_prob should be non negative.")
            if pepper_prob[1] < 0:
                raise ValueError("Upper pepper_prob should be non negative.")
            self.pepper_prob = pepper_prob
        elif isinstance(pepper_prob, (int, float)):
            if pepper_prob < 0:
                raise ValueError("pepper_prob should be non negative.")

            self.pepper_prob = (0, pepper_prob)
        else:
            raise TypeError(
                "Expected pepper_prob type to be one of (int, float, tuple, list), got {}".format(type(pepper_prob))
            )
            
    def apply(self, img, mask_salt=None, mask_pepper=None, **params):
        noisy_img = img.copy()
        noisy_img[mask_salt] = 255
        noisy_img[mask_pepper] = 0
        return noisy_img
    
    def get_params_dependent_on_targets(self, params):
        image = params["image"]
        apply_salt_prob = random.uniform(self.salt_prob[0], self.salt_prob[1])
        apply_pepper_prob = random.uniform(self.pepper_prob[0], self.pepper_prob[1])
        
        mask_salt = np.random.rand(*image.shape[:2]) < apply_salt_prob
        mask_pepper = np.random.rand(*image.shape[:2]) < apply_pepper_prob

        return {"mask_salt": mask_salt, "mask_pepper": mask_pepper}
    
    @property
    def targets_as_params(self):
        return ["image"]


def pixel_aug(p=0.2):
    pixel_transform = A.Compose(
        [
            A.ColorJitter(brightness=0.3, hue=0.2, p=p), # hue set to 0.1 for color sensitive model
            # A.OneOf(
            #     [
            #         A.RGBShift(), # not applicable
            #         A.ToGray(), # not applicable
            #     ],
            #     p=p
            # ),  # comment out for color sensitive model
            A.OneOf(
                [
                    # A.MedianBlur(blur_limit=3), # default setting for texture sensitive model
                    A.GaussianBlur(), # default setting for texture sensitive model
                    # A.ZoomBlur(max_factor=1.1), # not applicable
                    A.MotionBlur(allow_shifted=False), # default setting for texture sensitive model
                    # A.Defocus(),
                ],
                p=p
            ),
            A.OneOf(
                [
                    A.GaussNoise(per_channel=False), # var_limit default for texture sensitive model
                    # A.ISONoise(intensity=(0.1, 0.9)), # not applicable
                    # SaltPepperNoise(), # not applicable
                    A.ImageCompression(quality_lower=50, quality_upper=100),
                ],
                p=p
            )
        ]
    )
    return pixel_transform


def spatial_aug(p=0.4):
    spatial_transform = A.Compose(
        [
            A.Rotate(border_mode=cv.BORDER_CONSTANT, value=0, p=p), # 127 for cropped image, default is 0
            # A.SafeRotate(border_mode=cv.BORDER_CONSTANT, value=0, p=p), # safe rotate for detection
            A.VerticalFlip(p=p),
            A.HorizontalFlip(p=p),
            A.OneOf(
                [
                    A.ElasticTransform(),
                    A.GridDistortion(),
                    A.OpticalDistortion(),
                    A.Perspective(),    # comment out for pose sensitive model
                ],
                p=p
            )
        ],
    )
    return spatial_transform


def stereo_spatial_transform(img1, img2, disp, K, crop_size=None, 
                             spatial_aug_prob=0.5, do_flip="v", flip_prob=0.5):
    max_stretch = 0.2
    max_scale = 1.0
    stretch_prob = 0.8
    # randomly sample scale
    ht, wd = img1.shape[:2]
    if crop_size is None:
        crop_size = (int(ht * 2 // 3), int(wd * 2 // 3))
    if np.random.rand() < spatial_aug_prob:
        min_scale = np.maximum(
        (crop_size[0] + 8) / float(ht), 
        (crop_size[1] + 8) / float(wd))
        scale = np.random.uniform(min_scale, max_scale)
        scale_x = scale
        scale_y = scale
        if np.random.rand() < stretch_prob:
            scale_x *= 2 ** np.random.uniform(-max_stretch, max_stretch)
            scale_y *= 2 ** np.random.uniform(-max_stretch, max_stretch)
        
        scale_x = np.clip(scale_x, min_scale, None)
        scale_y = np.clip(scale_y, min_scale, None)
        
        # rescale the images
        img1 = cv.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv.INTER_LINEAR)
        img2 = cv.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv.INTER_LINEAR)
        disp = cv.resize(disp, None, fx=scale_x, fy=scale_y, interpolation=cv.INTER_LINEAR)

        disp = disp * scale_x
        K[0, 0] = K[0, 0] * scale_x
        K[1, 1] = K[1, 1] * scale_y

    if do_flip:
        # if np.random.rand() < self.h_flip_prob and self.do_flip == 'hf': # h-flip
        #     img1 = img1[:, ::-1]
        #     img2 = img2[:, ::-1]
        #     disp = disp[:, ::-1] * [-1.0, 1.0]

        # if np.random.rand() < self.h_flip_prob and self.do_flip == 'h': # h-flip for stereo
        #     tmp = img1[:, ::-1]
        #     img1 = img2[:, ::-1]
        #     img2 = tmp

        if np.random.rand() < flip_prob and do_flip == 'v': # v-flip
            img1 = img1[::-1, :]
            img2 = img2[::-1, :]
            disp = disp[::-1, :]

    # if self.yjitter:
    #     y0 = np.random.randint(2, img1.shape[0] - crop_size[0] - 2)
    #     x0 = np.random.randint(2, img1.shape[1] - crop_size[1] - 2)

    #     y1 = y0 + np.random.randint(-2, 2 + 1)
    #     img1 = img1[y0:y0+crop_size[0], x0:x0+crop_size[1]]
    #     img2 = img2[y1:y1+crop_size[0], x0:x0+crop_size[1]]
    #     disp = disp[y0:y0+crop_size[0], x0:x0+crop_size[1]]

    # else:
    y0 = np.random.randint(0, img1.shape[0] - crop_size[0])
    x0 = np.random.randint(0, img1.shape[1] - crop_size[1])
    
    img1 = img1[y0:y0+crop_size[0], x0:x0+crop_size[1]]
    img2 = img2[y0:y0+crop_size[0], x0:x0+crop_size[1]]
    disp = disp[y0:y0+crop_size[0], x0:x0+crop_size[1]]
    K[0, 2] = K[0, 2] - x0
    K[1, 2] = K[1, 2] - y0
    return img1, img2, disp, K

def get_resize_keep_aspect_ratio(H, W, divider=32, max_H=1344, max_W=1344):
    assert max_H % divider == 0
    assert max_W % divider == 0

    def round_by_divider(x):
        return int(np.ceil(x / divider) * divider)

    H_resize = round_by_divider(H)   #!NOTE KITTI width=1242
    W_resize = round_by_divider(W)
    if H_resize > max_H or W_resize > max_W:
        if H_resize > W_resize:
            W_resize = round_by_divider(W_resize * max_H / H_resize)
            H_resize = max_H
        else:
            H_resize = round_by_divider(H_resize * max_W / W_resize)
            W_resize = max_W
    return int(H_resize), int(W_resize)

class CustomDataset(Dataset):
    def __init__(self, list_path, data_aug=True, aug_prob=0.2,
                 spatial_aug_prob=1.0, slice=None, resize_scale=1,
                 replicate=False, rep_path=None, num_replicate=4):
        self.aug_flag = data_aug
        self.spatial_aug_prob = spatial_aug_prob
        self.slice = slice
        self.resize_scale = resize_scale
        self.input_size = (int(INPUT_H // self.resize_scale), int(INPUT_W // self.resize_scale))
        self.torch_transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Resize((self.input_size[0], self.input_size[1])),
        ])
        self.aug_prob = aug_prob
        self.pixel_aug = pixel_aug(p=aug_prob)
        self.spatial_aug = spatial_aug(p=aug_prob)
        
        # read data list
        self.path_data = []
        self.depth_path = []
        if type(list_path) != list:
            list_path = [list_path]
        for txt_path in list_path:
            with open(txt_path) as f:
                lines = f.readlines()
            if self.slice is not None:
                if self.slice < len(lines):
                    random.seed(2023)
                    lines = random.sample(lines, int(self.slice))
            for idx, line in enumerate(tqdm(lines)):
                splitted = line.strip().split(" ")
                self.path_data.append([])
                for i in range(len(splitted)):
                    path = splitted[i]
                    self.path_data[idx].append(path)
                    if "image_left" in path:
                        depth_file = path.replace("image_left/left", "depth_fsd/depth")
                        if not os.path.exists(depth_file):
                            print(f"Depth file not found: {depth_file}")
                            exit()
                        self.depth_path.append(depth_file)
            # replicate data
            if replicate and rep_path is not None:
                total_num = len(self.path_data)
                for idx in range(total_num):
                    for keyword in rep_path:
                        if keyword in self.path_data[idx][0]:
                            self.path_data = self.replicate_samples(
                                self.path_data, self.path_data[idx].copy(), num_replicate
                            )
                            self.label_data = self.replicate_samples(
                                self.label_data, self.label_data[idx].copy(), num_replicate
                            )
                            self.depth_path = self.replicate_samples(
                                self.depth_path, self.depth_path[idx], num_replicate
                            )
                print(f"Data replicate done. Before: {total_num}, After: {len(self.path_data)}")
                    
        # self.path_data = np.array(self.path_data, dtype=object)
        # self.label_data = np.array(self.label_data)
        
        # load camera parameters
        baseline = config.baseline
        focal_length = config.focal_length
        K = config.K
        self.camera_params = {
            "baseline": baseline,
            "focal_length": focal_length,
            "K": K
        }
        
    def __len__(self):
        return len(self.path_data)
    
    def replicate_samples(self, total_list: list, sample, replicate_num=5):
        for i in range(replicate_num):
            total_list.append(sample)
        return total_list

    def rescale_img_and_label(self, img, label):
        img = cv.resize(img, (self.input_size[1], self.input_size[0]))
        label = label // self.resize_scale
        return img, label

    def __getitem__(self, index):
        img_path_group = self.path_data[index]
        data_dict = {}
        img_list = []
        for i in range(len(img_path_group)):
            img_path = img_path_group[i]
            img = cv.imread(img_path)
            if img is None:
                print(img_path)
                
            if img.shape[0] != RAW_H or img.shape[1] != RAW_W:
                img = cv.resize(img, (RAW_W, RAW_H))
                
            # if self.aug_flag:
            #     transformed = self.pixel_aug(image=img)
            #     img = transformed["image"]
            #     transformed = self.spatial_aug(image=img)
            #     img = transformed["image"]

            # data_dict.setdefault("input", []).append(img)
            img_list.append(img)
            data_dict.setdefault("img_path", []).append(img_path)
        
        # load camera parameters
        baseline = self.camera_params["baseline"]
        focal_length = self.camera_params["focal_length"]
        K = self.camera_params["K"].copy()  # avoid shallow copy
        
        # load depth and disparity
        depth_file = self.depth_path[index]
        depth = cv.imread(depth_file, cv.IMREAD_UNCHANGED).astype(np.float32)
        depth_mask = (depth > 0)
        depth[~depth_mask] = 0
        
        disp = baseline * focal_length / (depth + 1e-6)
        # disp[disp > self.input_size[1]] = 0
        disp[~depth_mask] = 0
        
        # spatial transform
        if self.aug_flag and random.random() < self.aug_prob:
            img1, img2, disp, K = stereo_spatial_transform(
                img_list[0], img_list[1], disp, K,  
                spatial_aug_prob=self.spatial_aug_prob, do_flip=False
            )
        else:
            img1, img2 = img_list[0], img_list[1]
        
        crop_H, crop_W = img1.shape[:2]
        resize_H, resize_W = get_resize_keep_aspect_ratio(self.input_size[0], self.input_size[1])
        img1 = cv.resize(img1, (resize_W, resize_H))
        img2 = cv.resize(img2, (resize_W, resize_H))
        disp = cv.resize(disp, (resize_W, resize_H))
        disp = disp / (crop_W / resize_W)
        K[0, 0] = K[0, 0] / (crop_W / resize_W)
        K[1, 1] = K[1, 1] / (crop_H / resize_H)
        K[0, 2] = K[0, 2] / (crop_W / resize_W)
        K[1, 2] = K[1, 2] / (crop_H / resize_H)
        depth = baseline * K[0, 0] / np.clip(disp, 1, None)
        depth[depth < 0] = 0
        
        data_dict["baseline"] = np.float32(baseline)
        data_dict["focal_length"] = np.float32(K[0, 0])
        data_dict["K"] = np.float32(K)
        data_dict["depth"] = torch.from_numpy(depth).unsqueeze(0)
        data_dict["disp"] = torch.from_numpy(disp).unsqueeze(0)
        
        if self.aug_flag:
            transformed = self.pixel_aug(image=img1)
            img1 = transformed["image"]
            transformed = self.pixel_aug(image=img2)
            img2 = transformed["image"]
        
        img1 = self.torch_transform(img1)
        img2 = self.torch_transform(img2)
        data_dict["input"] = [img1, img2]
                     
        return data_dict['img_path'], img1, img2, data_dict["disp"], torch.ones_like(data_dict["disp"])


def fetch_loader(test_data_list="/mnt/ssd4/xingzenglan/libra/data_lists/zedx_val.list"):
    train_dataset = CustomDataset([test_data_list],
                       data_aug=False,
                        resize_scale=1)
    train_loader = DataLoader(train_dataset, batch_size=8, 
    pin_memory=True, shuffle=True, num_workers=8, drop_last=True)
    return train_loader

if __name__=='__main__':
    dataset = "sceneflow"
    dataset = "zedx"
    test_name = f"{dataset}_val_exp19"
    test_data_list = [f"/mnt/ssd4/xingzenglan/libra/data_lists/{dataset}_val.list"]
    ds = CustomDataset(test_data_list,
                       data_aug=False,
                        resize_scale=1)
    for i in range(10):
        sample = ds[i]
        pass