import math
import numpy as np
from enum import Enum
from typing import ( 
    List, 
    Tuple,
    Optional,
    Dict
)
import random
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F, InterpolationMode
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from batchgenerators.dataloading.data_loader import SlimDataLoaderBase
from batchgenerators.transforms.spatial_transforms import (
    SpatialTransform, 
    MirrorTransform
)
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.color_transforms import (
    BrightnessMultiplicativeTransform, 
    ContrastAugmentationTransform, 
    GammaTransform
)
from batchgenerators.transforms.utility_transforms import (
    RemoveLabelTransform, 
    RenameTransform, 
    NumpyToTensor
)
from batchgenerators.dataloading.data_loader import SlimDataLoaderBase
from batchgenerators.transforms.local_transforms import (
    BrightnessGradientAdditiveTransform,
    LocalGammaTransform,
    LocalSmoothingTransform,
    LocalContrastTransform
)
from batchgenerators.transforms.abstract_transforms import (
    Compose,
    AbstractTransform
)



class SingleImageMultiViewDataLoader(SlimDataLoaderBase):
    """Single image multi view dataloader.
    
    Adapted from batchgenerator examples:
    https://github.com/MIC-DKFZ/batchgenerators/blob/master/batchgenerators/examples/example_ipynb.ipynb
    """
    def __init__(
        self, 
        data: Dataset, 
        batch_size: int, 
        return_orig: str = True
    ):
        super(SingleImageMultiViewDataLoader, self).__init__(data, batch_size)
        self.return_orig = return_orig
    
    def generate_train_batch(self):
        # select single slice from dataset
        data = self._data[randrange(len(self._data))]
        # split into input and target, cast to np.float for batchgenerators
        img = data['input'].numpy().astype(np.float32)
        tar = data['target'][0].numpy().astype(np.float32)
        # copy along batch dimension
        img_batched = np.tile(img, (self.batch_size, 1, 1, 1))
        tar_batched = np.tile(tar, (self.batch_size, 1, 1, 1))
        # now construct the dictionary and return it. np.float32 cast because most networks take float
        out = {'data': img_batched, 
               'seg':  tar_batched}
        
        # if the original data is also needed, activate this flag to store it where augmentations
        # cant find it.
        if self.return_orig:
            out['data_orig']   = data['input'].unsqueeze(0)
            out['target_orig'] = data['target'].unsqueeze(0)
        
        return out
    
    
class MultiImageSingleViewDataLoader(SlimDataLoaderBase):
    """Multi image single view dataloader.
    
    Adapted from batchgenerator examples:
    https://github.com/MIC-DKFZ/batchgenerators/blob/master/batchgenerators/examples/example_ipynb.ipynb
    """
    def __init__(
        self, 
        data: Dataset, 
        batch_size: int, 
        return_orig: str = True
    ):
        super(MultiImageSingleViewDataLoader, self).__init__(data, batch_size)
        # data is now stored in self._data.
        self.return_orig = return_orig
    
    def generate_train_batch(self):
        # get random subset from dataset of batch size length
        sample = torch.randint(0, len(self._data), size=(self.batch_size,))
        data   = self._data[sample]
        # split into input and target
        img    = data['input']
        tar    = data['target']
        #construct the dictionary and return it. np.float32 cast because most networks take float
        out = {'data': img.numpy().astype(np.float32), 
               'seg':  tar.numpy().astype(np.float32)}
        
        # if the original data is also needed, activate this flag to store it where augmentations
        # cant find it.
        if self.return_orig:
            out['data_orig']   = img
            out['target_orig'] = tar
        
        return out
    
    
class Transforms(object):
    """
    A container for organizing and accessing different sets of image transformation operations.

    This class defines four categories of transformations: 'io_transforms', 
    'global_nonspatial_transforms', 'global_transforms', and 'local_transforms'. Each category 
    contains a list of specific transform objects designed for image preprocessing in machine learning tasks.

    The 'io_transforms' are basic input/output operations, like renaming and format conversion.
    'global_nonspatial_transforms' apply transformations like noise addition or resolution simulation 
    that do not depend alter spatial information. 'global_transforms' include spatial transformations 
    like rotation and scaling. 'local_transforms' focus on localized changes to an image, such as adding 
    brightness gradients or local smoothing and are also non-spatial.

    Attributes:
        transforms (dict): A dictionary where keys are transform categories and values are lists of transform objects.

    Methods:
        get_transforms(arg: str): Retrieves a composed transform pipeline based on the specified category.

    Usage:
        >>> transforms = Transforms()
        >>> global_transforms = transforms.get_transforms('global_transforms')
    """
    
    def __init__(
        self,
    ) -> None:
        
        io_transforms = [
            RemoveLabelTransform(
                output_key = 'seg', 
                input_key = 'seg',
                replace_with = 0,
                remove_label = -1
            ),
           RenameTransform(
                delete_old = True,
                out_key = 'target',
                in_key = 'seg'
            ),
           NumpyToTensor(
                keys = ['data', 'target'], 
                cast_to = 'float')    
        ]
       
        global_nonspatial_transforms = [
            SimulateLowResolutionTransform(
                order_upsample = 3, 
                order_downsample = 0, 
                channels = None, 
                per_channel = True, 
                p_per_channel = 0.5, 
                p_per_sample = 0.25, 
                data_key = 'data',
                zoom_range = (0.5, 1), 
                ignore_axes = None
            ),
            GaussianNoiseTransform(
                p_per_sample = 0.1, 
                data_key = 'data', 
                noise_variance = (0, 0.1), 
                p_per_channel = 1, 
                per_channel = False
            ),
        ] 
       
        global_transforms = [
            SpatialTransform(
                independent_scale_for_each_axis = False, 
                p_rot_per_sample = 0.2, 
                p_scale_per_sample = 0.2, 
                p_el_per_sample = 0.2, 
                data_key = 'data', 
                label_key = 'seg', 
                patch_size = np.array([256, 256]), 
                patch_center_dist_from_border = None, 
                do_elastic_deform = False, 
                alpha = (0.0, 200.0), 
                sigma = (9.0, 13.0), 
                do_rotation = True, 
                angle_x = (-3.141592653589793, 3.141592653589793), 
                angle_y = (-0.0, 0.0), 
                angle_z = (-0.0, 0.0), 
                do_scale = True,
                scale = (0.7, 1.4), 
                border_mode_data = 'constant',
                border_cval_data = 0, 
                order_data = 3, 
                border_mode_seg = 'constant',
                border_cval_seg = -1, 
                order_seg = 1,
                random_crop = False,
                p_rot_per_axis = 1, 
                p_independent_scale_per_axis = 1
            ),
            GaussianBlurTransform(
                p_per_sample = 0.2, 
                different_sigma_per_channel = True, 
                p_per_channel = 0.5, 
                data_key = 'data', 
                blur_sigma = (0.5, 1.0), 
                different_sigma_per_axis = False, 
                p_isotropic = 0
            ),
            BrightnessMultiplicativeTransform(
                p_per_sample = 0.15, 
                data_key = 'data', 
                multiplier_range = (0.75, 1.25), 
                per_channel = True
            ),
            ContrastAugmentationTransform(
                p_per_sample = 0.15, 
                data_key = 'data', 
                contrast_range = (0.75, 1.25), 
                preserve_range = True, 
                per_channel = True, 
                p_per_channel = 1
            ),
            GammaTransform(
                p_per_sample = 0.1,
                retain_stats = True, 
                per_channel = True, 
                data_key = 'data', 
                gamma_range = (0.7, 1.5), 
                invert_image = True
            ),
            GammaTransform(
                p_per_sample = 0.3,
                retain_stats = True, 
                per_channel = True, 
                data_key = 'data', 
                gamma_range = (0.7, 1.5), 
                invert_image = False
            ),
            MirrorTransform(
                p_per_sample = 1, 
                data_key = 'data', 
                label_key = 'seg', 
                axes = (0, 1)
            ),
        ]       
        
        
        local_transforms = [
            BrightnessGradientAdditiveTransform(
                scale=200, 
                max_strength=4, 
                p_per_sample=0.2, 
                p_per_channel=1
            ),
            LocalGammaTransform(
                scale=200, 
                gamma=(2, 5), 
                p_per_sample=0.2,
                p_per_channel=1
            ),
            LocalSmoothingTransform(
                scale=200,
                smoothing_strength=(0.5, 1),
                p_per_sample=0.2,
                p_per_channel=1
            ),
            LocalContrastTransform(
                scale=200,
                new_contrast=(1, 3),
                p_per_sample=0.2,
                p_per_channel=1
            ),
        ]
        
        self.transforms = {
            'io_transforms': io_transforms,
            'global_nonspatial_transforms': global_nonspatial_transforms + io_transforms,
            'global_transforms': global_transforms + io_transforms,
            'local_transforms': global_nonspatial_transforms + local_transforms + io_transforms,
        }


    def get_transforms(
        self, 
        arg: str
    ) -> AbstractTransform:
        return Compose(self.transforms[arg])



def _apply_op(
    img: Tensor,
    op_name: str,
    magnitude: float,
    interpolation: InterpolationMode,
    fill: Optional[List[float]],
):
    if op_name == "ShearX":
        # magnitude should be arctan(magnitude)
        # official autoaug: (1, level, 0, 0, 1, 0)
        # https://github.com/tensorflow/models/blob/dd02069717128186b88afa8d857ce57d17957f03/research/autoaugment/augmentation_transforms.py#L290
        # compared to
        # torchvision:      (1, tan(level), 0, 0, 1, 0)
        # https://github.com/pytorch/vision/blob/0c2373d0bba3499e95776e7936e207d8a1676e65/torchvision/transforms/functional.py#L976
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[math.degrees(math.atan(magnitude)), 0.0],
            interpolation=interpolation,
            fill=fill,
            center=[0, 0],
        )
    elif op_name == "ShearY":
        # magnitude should be arctan(magnitude)
        # See above
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[0.0, math.degrees(math.atan(magnitude))],
            interpolation=interpolation,
            fill=fill,
            center=[0, 0],
        )
    elif op_name == "TranslateX":
        img = F.affine(
            img,
            angle=0.0,
            translate=[int(magnitude), 0],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif op_name == "TranslateY":
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, int(magnitude)],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif op_name == "Zoom":
        scale = (100 - magnitude) / 100
        img = F.resized_crop(
            img,
            top=int(img.shape[-2] // 2 - (img.shape[-2] * scale) // 2),
            left=int(img.shape[-1] // 2 - (img.shape[-1] * scale) // 2),
            height=int(img.shape[-2] * scale),
            width=int(img.shape[-1] * scale),
            size=img.shape[-2:],
        )
    elif op_name == "Rotate":
        img = F.rotate(img, magnitude, interpolation=interpolation, fill=fill)
    elif op_name == "Brightness":
        img = F.adjust_brightness(img, 1.0 + magnitude)
    elif op_name == "Color":
        img = F.adjust_saturation(img, 1.0 + magnitude)
    elif op_name == "Contrast":
        img = F.adjust_contrast(img, 1.0 + magnitude)
    elif op_name == "Sharpness":
        img = F.adjust_sharpness(img, 1.0 + magnitude)
    elif op_name == "Posterize":
        img = F.posterize(img, int(magnitude))
    elif op_name == "Solarize":
        img = F.solarize(img, magnitude)
    elif op_name == "AutoContrast":
        img = F.autocontrast(img)
    elif op_name == "Equalize":
        img = F.equalize(img)
    elif op_name == "Invert":
        img = F.invert(img)
    elif op_name == "Identity":
        pass
    else:
        raise ValueError(f"The provided operator {op_name} is not recognized.")
    return img



class RandAugmentWithLabels(torch.nn.Module):
    r"""RandAugment data augmentation method based on
    `"RandAugment: Practical automated data augmentation with a reduced search space"
    <https://arxiv.org/abs/1909.13719>`_.
    If the image is torch Tensor, it should be of type torch.uint8, and it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Args:
        num_max_ops (int): Maximum number of augmentation transformations to apply sequentially.
        magnitude (int): Magnitude for all the transformations.
        num_magnitude_bins (int): The number of different magnitude values.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
        fill (sequence or number, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.
    """

    def __init__(
        self,
        num_max_ops: int = 2,
        magnitude: int = 9,
        num_magnitude_bins: int = 31,
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        fill: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        self.num_max_ops = num_max_ops
        self.magnitude = magnitude
        self.num_magnitude_bins = num_magnitude_bins
        self.interpolation = interpolation
        self.fill = fill

    def _augmentation_space(
        self, num_bins: int, image_size: List[int]
    ) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            # op_name: (magnitudes, signed)
            "Identity": (torch.tensor(0.0), False),
            "ShearX": (torch.linspace(0.0, 0.3, num_bins), True),
            "ShearY": (torch.linspace(0.0, 0.3, num_bins), True),
            "TranslateX": (
                torch.linspace(0.0, 150.0 / 331.0 * image_size[0], num_bins),
                True,
            ),
            "TranslateY": (
                torch.linspace(0.0, 150.0 / 331.0 * image_size[1], num_bins),
                True,
            ),
            "Rotate": (torch.linspace(0.0, 45.0, num_bins), True),
            "Zoom": (torch.linspace(0.0, 50.0, num_bins), False),
            "Brightness": (torch.linspace(0.0, 0.9, num_bins), True),
            "Color": (torch.linspace(0.0, 0.9, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.9, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.9, num_bins), True),
            "Posterize": (
                8 - (torch.arange(num_bins) / ((num_bins - 1) / 4)).round().int(),
                False,
            ),
            "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }

    def forward(self, img: Tensor) -> Tensor:
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Transformed image.
        """
        fill = self.fill
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * F.get_image_num_channels(img)
            elif fill is not None:
                fill = [float(f) for f in fill]

        op_trace = []

        # for _ in range(torch.randint(0, self.num_max_ops, (1,))):
        for _ in range(random.randint(0, self.num_max_ops)):
            op_meta = self._augmentation_space(
                self.num_magnitude_bins, F.get_image_size(img)
            )
            op_index = int(torch.randint(len(op_meta), (1,)).item())
            op_name = list(op_meta.keys())[op_index]
            magnitudes, signed = op_meta[op_name]
            magnitude = (
                float(magnitudes[self.magnitude].item()) if magnitudes.ndim > 0 else 0.0
            )
            magnitude = torch.rand((1,)).item() * magnitude
            if signed and torch.randint(2, (1,)):
                magnitude *= -1.0
            img = _apply_op(
                img, op_name, magnitude, interpolation=self.interpolation, fill=fill
            )
            if op_name in [
                "Identity",
                "ShearX",
                "ShearY",
                "TranslateX",
                "TranslateY",
                "Rotate",
                "Zoom",
            ]:
                op_trace.append((op_name, magnitude, InterpolationMode.NEAREST, fill))
                # op_trace.append((op_name, magnitude, self.interpolation, fill))
                # print("label transformed")
                # lbl = _apply_op(lbl, op_name, magnitude, interpolation=self.interpolation, fill=fill)

        return img, op_trace

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"num_max_ops={self.num_max_ops}"
            f", magnitude={self.magnitude}"
            f", num_magnitude_bins={self.num_magnitude_bins}"
            f", interpolation={self.interpolation}"
            f", fill={self.fill}"
            f")"
        )
        return s

        
def volume_collate(batch: List[dict]) -> dict:
    return batch[0]

        
def slice_selection(
    dataset: Dataset, 
    indices: Tensor,
    n_cases: int = 10
) -> Tensor:
    
    slices = dataset.__getitem__(indices)['input']
    kmeans_in = slices.reshape(len(indices), -1)
    kmeans = KMeans(n_clusters=n_cases).fit(kmeans_in)
    idx, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, kmeans_in)
    return indices[idx]


def dataset_from_indices(
    dataset: Dataset, 
    indices: Tensor
) -> DataLoader:
    
    data = dataset.__getitem__(indices)
    
    class CustomDataset(Dataset):
        
        def __init__(self, input: Tensor, labels: Tensor, 
                     voxel_dim: Tensor):
            self.input = input
            self.labels = labels
            self.voxel_dim = voxel_dim
            
        def __getitem__(self, idx):
            return {'input': self.input[idx],
                    'target': self.labels[idx],
                    'voxel_dim': self.voxel_dim[idx]}
        
        def __len__(self):
            return self.input.size(0)
        
    return CustomDataset(*data.values())


@torch.no_grad()
def get_subset(
    dataset: Dataset,
    model: UNet2D,
    criterion: nn.Module,
    device: str = 'cuda:0',
    fraction: float = 0.1, 
    n_cases: int = 10, 
    part: str = "tail",
    batch_size: int = 1
) -> Dataset:
    """Selects a subset of the otherwise very large CC-359 Dataset, which
        - is in the bottom/top fraction w.r.t. to a criterion and model
        - contains n_cases, drawn to be divers w.r.t. to the input space and
          defined as k_means cluster centers, one for each case.
    """
    # TODO: cache subset indices for subsequent runs. Cache based on all
    # factors that influence selection, i.e. model, criterion, function params etc
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,
        drop_last=False
    )
    
    # collect evaluation per slice and cache
    assert criterion.reduction == 'none'
    model.eval()
    loss_list = []
    for batch in dataloader:
        input_  = batch['input'].to(device)
        target  = batch['target'].to(device)
        net_out = model(input_)
        loss    = criterion(net_out, target).view(input_.shape[0], -1).mean(1)
        loss_list.append(loss)
        
    loss_tensor = torch.cat(loss_list)
    assert len(loss_tensor.shape) == 1
    
    # get indices from loss function values
    indices = torch.argsort(loss_tensor, descending=True)
    # set some params for slicing
    len_ = len(dataset)
    devisor = int(1 / fraction)
    # Chunk dataset for either tail or head part based on func args
    if part == 'tail':
        indices = indices[:len_ // devisor]
    elif part == 'head':
        indices = indices[-len_ // devisor:]   
    # select slices within fraction of dataset based on kmeans
    indices_selection = slice_selection(dataset, indices, n_cases=n_cases)
    # build dataset from selected slices and return
    subset = dataset_from_indices(dataset, indices_selection)

    return subset