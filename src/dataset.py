"""
This script collects the Dataset classes for the CC-359, 
ACDC and M&M data sets.

Usage: serves only as a collection of individual functionalities
Authors: Rasha Sheikh, Jonathan Lennartz
"""


# - standard packages
import os

# - third party packages
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import (
    Resize,
    CenterCrop,
    Normalize,
    functional,
)
from sklearn.preprocessing import MinMaxScaler
import nibabel as nib
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.paths import preprocessing_output_dir
from nnunet.training.dataloading.dataset_loading import *

# - local source
from augment import RandAugmentWithLabels, _apply_op


class CalgaryCampinasDataset(Dataset):
    def __init__(
        self,
        data_path,
        site=2,
        augment=False,
        normalize=False,
        volume_wise=False,
        split="all",
        debug=False,
        seed=True,
    ):
        #         Training
        #         Vendor 6
        #         Length of dataset: 12096
        #         Validation
        #         Vendor 6
        #         Length of dataset: 1344
        #         Vendor 1
        #         Length of dataset: 15360
        #         Vendor 2
        #         Length of dataset: 11792
        #         Vendor 3
        #         Length of dataset: 9201
        #         Vendor 4
        #         Length of dataset: 10800
        #         Vendor 5
        #         Length of dataset: 11520

        assert site in [1,2,3,4,5,6], 'select a valid site'

        if site == 1:
            self.folder = "GE_15"
        elif site == 2:
            self.folder = "GE_3"
        elif site == 3:
            self.folder = "Philips_15"
        elif site == 4:
            self.folder = "Philips_3"
        elif site == 5:
            self.folder = "Siemens_15"
        elif site == 6:
            self.folder = "Siemens_3"
        else:
            print("No site specified!")
            self.folder = "GE_3"

        self.split = split
        self.debug = debug
        self.seed = seed
        self.normalize = normalize
        self.augment = augment

        self.crop = CenterCrop((256, 256))

        self.volume_wise = volume_wise

        if self.normalize:
            self.normalize = Normalize(mean=[0.0912], std=[0.1280])

        if self.augment:
            self.rand_augment = RandAugmentWithLabels(
                num_max_ops=3,
                magnitude=9,
                interpolation=functional.InterpolationMode.BILINEAR,
            )

        self.load_files(data_path)

    def pad_image(self, img):
        s, h, w = img.shape
        if h < w:
            b = (w - h) // 2
            a = w - (b + h)
            return np.pad(img, ((0, 0), (b, a), (0, 0)), mode="edge")
        elif w < h:
            b = (h - w) // 2
            a = h - (b + w)
            return np.pad(img, ((0, 0), (0, 0), (b, a)), mode="edge")
        else:
            return img

    def pad_image_w_size(self, data_array, max_size):
        current_size = data_array.shape[-1]
        b = (max_size - current_size) // 2
        a = max_size - (b + current_size)
        return np.pad(data_array, ((0, 0), (b, a), (b, a)), mode="edge")

    def crop_center(self, img, cropw=256, croph=256):
        _, w, h = img.shape
        if w == cropw and h == croph:
            return img
        else:
            startw = w // 2 - (cropw // 2)
            starth = h // 2 - (croph // 2)
            return img[:, startw:startw + cropw, starth:starth + croph]

    def unify_sizes(self, input_images, input_labels):
        sizes = np.zeros(len(input_images), int)
        for i in range(len(input_images)):
            sizes[i] = input_images[i].shape[-1]
        max_size = np.max(sizes)

        for i in range(len(input_images)):
            if sizes[i] != max_size:
                input_images[i] = self.pad_image_w_size(input_images[i],
                        max_size)
                input_labels[i] = self.pad_image_w_size(input_labels[i],
                        max_size)
        return input_images, input_labels

    def load_files(self, data_path):
        scaler = MinMaxScaler()

        images = []
        labels = []
        volume_ranges = [0]
        self.voxel_dim = []

        images_path = os.path.join(data_path, "Original", self.folder)
        files = np.array(sorted(os.listdir(images_path)))

        if self.split != "all":
            if self.seed:
                np.random.seed(42)

            indices = np.random.permutation(len(files))
            split = len(files) - len(files) // 10

            if self.split == "train":
                files = files[indices[:split]]
            elif self.split == "validation":
                files = files[indices[split:]]

        for i, f in enumerate(files):
            if i == 3 and self.debug:
                break

            nib_file = nib.load(os.path.join(images_path, f))
            img = nib_file.get_fdata("unchanged", dtype=np.float32)

            lbl = nib.load(
                os.path.join(
                    data_path, "Silver-standard", self.folder, f[:-7] + "_ss.nii.gz"
                )
            ).get_fdata("unchanged", dtype=np.float32)

            transformed = scaler.fit_transform(np.reshape(img, (-1, 1)))
            img = np.reshape(transformed, img.shape)
            img = np.rot90(img, axes=(1, 2))

            if img.shape[1] != img.shape[2]:
                img = self.pad_image(img)

            volume_ranges.append(img.shape[0])
            images.append(img)

            lbl = np.rot90(lbl, axes=(1, 2))
            if lbl.shape[1] != lbl.shape[2]:
                lbl = self.pad_image(lbl)

            labels.append(lbl)
            spacing = [nib_file.header.get_zooms()] * img.shape[0]
            self.voxel_dim.append(np.array(spacing))

        self.volume_ranges = np.cumsum(np.array(volume_ranges))

        images, labels = self.unify_sizes(images, labels)

        self.data = np.expand_dims(np.vstack(images), axis=1)
        self.label = np.expand_dims(np.vstack(labels), axis=1)
        self.voxel_dim = np.vstack(self.voxel_dim)

        self.data = torch.from_numpy(self.data)
        if self.normalize:
            self.data = self.normalize(self.data)

        self.label = torch.from_numpy(self.label)
        self.voxel_dim = torch.from_numpy(self.voxel_dim)

        self.min = self.data.min()
        self.max = self.data.max()

    def __len__(self):
        if self.volume_wise:
            return len(self.volume_ranges) - 1
        else:
            return len(self.data)

    def __getitem__(self, idx):
        if self.volume_wise:
            data = self.data[self.volume_ranges[idx]:self.volume_ranges[idx + 1]]
            labels = self.label[self.volume_ranges[idx]:self.volume_ranges[idx + 1]]
            voxel_dim = self.voxel_dim[
                self.volume_ranges[idx]:self.volume_ranges[idx + 1]
            ]
        else:
            data = self.data[idx]
            labels = self.label[idx]
            voxel_dim = self.voxel_dim[idx]

        if self.augment:
            data = 255 * (data - self.min) / (self.max - self.min)
            data = data.type(torch.uint8).contiguous()
            data, op_trace = self.rand_augment(data)
            data = data.float()
            data = (data / 255.0) * (self.max - self.min) + self.min

            for op in op_trace:
                labels = _apply_op(labels, *op)

        data = self.crop(data)
        labels = self.crop(labels)

        return {"input": data, "target": labels, "voxel_dim": voxel_dim}


class ACDCDataset(Dataset):
    def __init__(
        self, 
        data: str = "train", # or 'val'
        debug: bool = False,
        root: str = '../../../',
        folder: str = 'nnUNet/data/nnUNet_preprocessed/Task500_ACDC/',
    ):  
        self.data = data
        self.debug = debug
        self.root = root
        self.folder = folder
        self.crop = CenterCrop([256, 256])
        self.resize = Resize((256, 224))
        self._get_dataset_information()
        self._load_selected_cases()

    def _get_dataset_information(self) -> None:
        # we only want to load M&M challenge data with this class. The following code is from
        # the nnUNet repo and simply loads the dataset information. We use this information to
        # filter the data later and load the appropriate files corresponding to e.g. a
        # specific vendor.
        t = "Task500_ACDC"
        p = join(preprocessing_output_dir, t, "nnUNetData_plans_v2.1_2D_stage0")
        self.dataset_info = load_dataset(p)
        with open(
            join(join(preprocessing_output_dir, t), "nnUNetPlansv2.1_plans_2D.pkl"),
            "rb",
        ) as f:
            plans = pickle.load(f)
        unpack_dataset(p)
        # select keys that are relevant for a specific vendor
        with open(
            f"{self.root}{self.folder}splits_final.pkl",
            "rb",
        ) as f:
            self.plans = pickle.load(f)
        self.train_files = list(self.plans[0][self.data])
        self.keys = [
            key for key in list(self.dataset_info.keys()) if key in self.train_files
        ]

    def _load_selected_cases(self) -> None:
        self.data = []
        for key in self.keys:
            # load data as numpy array from nnUNet preprocessed folder
            data_np = np.load(self.dataset_info[key]["data_file"][:-4] + ".npz")["data"]
            # Transform to torch tensor and append to list.
            data = torch.from_numpy(data_np)
            # Crop data to ACDC data shape, i.e. (256, 224).
            data = self.crop(data)
            self.data.append(data)
        # cat list to single tensor
        self.data = torch.cat(self.data, dim=1).unsqueeze(2)
        # for debugging purposes only take the first 50 cases
        if self.debug:
            self.data = self.data[:, :50]
        # split data into input and target
        self.input = self.data[0]
        self.target = self.data[1]

    def __len__(self):
        return self.data.shape[1]

    def __getitem__(self, idx):
        return {
            "input": self.input[idx],
            "target": self.target[idx],
            "voxel_dim": torch.tensor([1.0, 1.0, 1.0]),
        }


class MNMDataset(Dataset):
    # 1: Siemens
    # 2: Philips
    # 3: GE
    # 4: Canon

    def __init__(
        self, 
        vendor: str, 
        debug: bool = False, 
        selection: str = 'all_cases', 
#         adapt_size: str = 'crop'
    ):
        assert vendor in ['A', 'B', 'C', 'D'], 'select suitable vendor'
        self.vendor = vendor
        self.debug = debug
        self.selection = selection
#         if adapt_size == "crop":
        self.crop = CenterCrop([256, 256])
#         elif adapt_size == "resize":
#             self.resize = Resize((256, 224))
        self._get_dataset_information()
        self._load_selected_cases()

    def _get_dataset_information(self) -> None:
        # we only want to load M&M challenge data with this class. The following code is from
        # the nnUNet repo and simply loads the dataset information. We use this information to
        # filter the data later and load the appropriate files corresponding to e.g. a
        # specific vendor.
        t = "Task679_heart_mnms"  # "old/Task679_mnm" #
        p = join(preprocessing_output_dir, t, "nnUNetData_plans_v2.1_2D_stage0")
        self.dataset_info = load_dataset(p)
        with open(
            join(join(preprocessing_output_dir, t), "nnUNetPlansv2.1_plans_2D.pkl"),
            "rb",
        ) as f:
            plans = pickle.load(f)
        unpack_dataset(p)
        # select keys that are relevant for a specific vendor
        self.vendor_keys = [
            key for key in list(self.dataset_info.keys()) if f"_{self.vendor}" in key
        ]

    def _load_selected_cases(self) -> None:
        self.data = []
        counter = 0
        for key in self.vendor_keys:
            # load data as numpy array from nnUNet preprocessed folder
            data_np = np.load(self.dataset_info[key]["data_file"][:-4] + ".npz")["data"]
            # Transform to torch tensor and append to list.
            data = torch.from_numpy(data_np)
            # Crop data to ACDC data shape, i.e. (256, 224).
            data = self.crop(data)
            # merge background classes -1 and 0
            data[1][data[1] < 0] = 0
            assert (data[1] < 0).sum() == 0
            
            # mask slices with empty targets
            if self.selection == 'non_empty_target':
                mask = data[1].sum((1,2)) > 0
            
            # keep only slices that containt all 4 classes
            if self.selection == 'all_classes':
                mask = torch.zeros((data.shape[1]), dtype=bool)
                for i, slc in enumerate(data[1]):
                    mask[i] = len(torch.unique(slc)) >= 4
            
            # keep only slices that contain 3 out of the 4 classes
            if self.selection == 'single_class_missing':
                mask = torch.zeros((data.shape[1]), dtype=bool)
                for i, slc in enumerate(data[1]):
                    mask[i] = len(torch.unique(slc)) == 3

            # dont mask if mask is none
            if self.selection == 'all_cases':
                pass
            else:
                assert len(mask.shape) == 1
                data = data[:, mask]

            self.data.append(data)

        # cat list to single tensor
        self.data = torch.cat(self.data, dim=1).unsqueeze(2)
        
        # for debugging purposes only take the first 50 cases
        if self.debug:
            self.data = self.data[:, :50]
        # split data into input and target
        self.input  = self.data[0]
        self.target = self.data[1]
        # swap values of 3 and 1 in target so that its
        # similar to the ACDC data convention
        self.target[self.target == 1] = 999
        self.target[self.target == 3] = 1
        self.target[self.target == 999] = 3
#         # merge background classes -1 and 0
#         self.target[self.target < 0] = 0
        

    def __len__(self):
        return self.data.shape[1]

    def __getitem__(self, idx):
        return {
            "input": self.input[idx],
            "target": self.target[idx],
            "voxel_dim": torch.tensor([1.0, 1.0, 1.0]),
        }
