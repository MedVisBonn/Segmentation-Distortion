import torch
from torch import nn, Tensor
from typing import Iterable, Dict, Callable, Tuple, Union
from copy import deepcopy
from model.unet import UNet2D



class ModelAdapter(nn.Module):
    """Wrapper class for segmentation models and feature transformations.

    Wraps (a copy of) the segmentation model and attaches feature
    trasformations to its swivels via hooks (at potentially various positions
    simultaneously). Additionally, it provides control utilities for the
    hooks as well as different types for inference training and inspection.
    """

    def __init__(
        self,
        seg_model: nn.Module,
        transformations: nn.ModuleDict,
        disabled_ids: list = [],
        copy: bool = True,
    ):
        super().__init__()
        self.seg_model = deepcopy(seg_model) if copy else seg_model

        self.transformations = transformations
        self.disabled_ids = disabled_ids
        self.transformation_handles = {}
        self.train_transformation_handles = {}
        self.inspect_transformation_handles = {}
        self.training_data = {}
        self.inspect_data = {}


    def hook_train_transformations(
        self, 
        transformations: Dict[str, nn.Module]
    ) -> None:
        for layer_id in transformations:
            layer = self.seg_model.get_submodule(layer_id)
            hook = self._get_train_transformation_hook(
                transformations[layer_id], layer_id
            )
            self.train_transformation_handles[
                layer_id
            ] = layer.register_forward_pre_hook(hook)


    def hook_inference_transformations(
        self, transformations: Dict[str, nn.Module], 
        n_samples: int
    ) -> None:
        for layer_id in transformations:
            layer = self.seg_model.get_submodule(layer_id)
            hook = self._get_inference_transformation_hook(
                transformations[layer_id],
                n_samples
            )
            self.transformation_handles[layer_id] = layer.register_forward_pre_hook(
                hook
            )
            

    def hook_inspect_transformation(
        self, 
        transformations: Dict[str, nn.Module], 
    ) -> None:
        for layer_id in transformations:
            if layer_id not in self.disabled_ids:
                layer = self.seg_model.get_submodule(layer_id)
                hook  = self._get_inspect_transformation_hook(transformations[layer_id], layer_id)
                self.inspect_transformation_handles[
                    layer_id
                ] = layer.register_forward_pre_hook(hook)


    def _get_train_transformation_hook(
        self,
        transformation: nn.Module,
        layer_id: str
    ) -> Callable:
        def hook(module: nn.Module, x: Tuple[Tensor]) -> Tensor:
            x_in, *_ = x  # tuple, alternatively use x_in = x[0]
            #print(x_in.shape, x_in.shape[0] // 2)
            batch_size = x_in.shape[0] // 2
            x_orig, _  = torch.split(x_in, batch_size)
            x_in_denoised = transformation(x_in)
            
            if layer_id not in self.disabled_ids:
                mse = nn.functional.mse_loss(
                    x_in_denoised, 
                    x_orig.repeat(2,1,1,1).detach(),
                    reduction="mean"
                )

                training_data = {
                    "mse": mse,
                }

                self.training_data[layer_id] = training_data

            return torch.cat([x_orig, x_in_denoised[batch_size:]], dim=0)
            
        return hook
    

    def _get_inference_transformation_hook(
        self, transformation: nn.Module, n_samples: int = 1
    ) -> Callable:
        def hook(module: nn.Module, x: Tuple[Tensor]) -> Tensor:
            x_in, *_ = x  # weird tuple, can use x_in = x[0]
            if n_samples == 0:
                return x
            elif n_samples == -1:
                x_in_new = transformation(x_in)
                return x_in_new
            else:
                x_in_new = x_in.unsqueeze(1).repeat(1, n_samples, 1, 1, 1).flatten(0, 1)
                x_in_new = transformation(x_in_new)
                return torch.cat([x_in, x_in_new], dim=0)

        return hook
    
    
    def _get_inspect_transformation_hook(
            self, 
            transformation: nn.Module, 
            layer_id: str, 
        ) -> Callable:
        
        @torch.no_grad()
        def hook(module: nn.Module, x: Tuple[Tensor]) -> Tensor:
            x_in, *_ = x  # weird tuple, can use x_in = x[0]
            x_orig = x_in[:1]
            x_in_denoised = transformation(x_in)
            residuals     = x_in_denoised - x_in
                
            if layer_id not in self.disabled_ids:
                data = {
                    'input'     : x_in,
                    'denoised'  : x_in_denoised,
                    'residuals' : residuals
                }
                
                self.inspect_data[layer_id] = data
            
            return torch.cat([x_orig, x_in_denoised[1:]], dim=0)
        
        return hook
    

    def remove_train_transformation_hook(self, layer_id: str) -> None:
        self.train_transformation_handles[layer_id].remove()

    def remove_transformation_hook(self, layer_id: str) -> None:
        self.transformation_handles[layer_id].remove()
        
    def remove_inspect_transformation_hook(self, layer_id: str) -> None:
        self.inspect_transformation_handles[layer_id].remove()

    def remove_all_hooks(self):
        if hasattr(self, "train_transformation_handles"):
            for handle in self.train_transformation_handles:
                self.train_transformation_handles[handle].remove()
            self.train_transformation_handles = {}

        if hasattr(self, "transformation_handles"):
            for handle in self.transformation_handles:
                self.transformation_handles[handle].remove()
            self.transformation_handles = {}
            
        if hasattr(self, 'inspect_transformation_handles'):
            for handle in self.inspect_transformation_handles:
                self.inspect_transformation_handles[handle].remove()
            self.inspect_transformation_handles = {}
        

    def freeze_seg_model(self):
        self.seg_model.eval()
        for param in self.seg_model.parameters():
            param.requires_grad = False

    def set_number_of_samples_to(self, n_samples: int):
        self.n_samples = n_samples

    def disable(self, layer_ids: list) -> None:
        for layer_id in layer_ids:
            self.transformations[layer_id].turn_off()

    def enable(self, layer_ids: list) -> None:
        for layer_id in layer_ids:
            self.transformations[layer_id].turn_on()

    def forward(self, x: Tensor):
        return self.seg_model(x)


# class Frankenstein(nn.Module):
#     """Wrapper class for segmentation models and feature transformations.

#     Wraps (a copy of) the segmentation model and attaches feature
#     trasformations to it via hooks (at potentially various positions
#     simultaneously). Additionally, it provides control utilities for the
#     hooks as well as different types for inference and training.
#     """

#     def __init__(
#         self,
#         seg_model: nn.Module,
#         transformations: nn.ModuleDict,
#         disabled_ids: list = [],
#         copy: bool = True,
#     ):
#         super().__init__()
#         self.seg_model = deepcopy(seg_model) if copy else seg_model

#         self.transformations = transformations
#         self.disabled_ids = disabled_ids
#         self.transformation_handles = {}
#         self.train_transformation_handles = {}
#         self.inspect_transformation_handles = {}
#         self.training_data = {}
#         self.inspect_data = {}

#     def hook_train_transformations(self, transformations: Dict[str, nn.Module]) -> None:
#         for layer_id in transformations:
#             layer = self.seg_model.get_submodule(layer_id)
#             hook = self._get_train_transformation_hook(
#                 transformations[layer_id], layer_id
#             )
#             self.train_transformation_handles[
#                 layer_id
#             ] = layer.register_forward_pre_hook(hook)

#     def hook_inference_transformations(
#         self, transformations: Dict[str, nn.Module], n_samples: int
#     ) -> None:
#         for layer_id in transformations:
#             layer = self.seg_model.get_submodule(layer_id)
#             hook = self._get_inference_transformation_hook(transformations[layer_id], n_samples)
#             self.transformation_handles[layer_id] = layer.register_forward_pre_hook(hook)
            
#     def hook_inspect_transformation(
#         self, 
#         transformations: Dict[str, nn.Module], 
#         n_samples: int,
#         arch: str = 'ae'
#     ) -> None:
#         for layer_id in transformations:
#             if layer_id not in self.disabled_ids:
#                 layer = self.seg_model.get_submodule(layer_id)
#                 hook  = self._get_inspect_transformation_hook(transformations[layer_id], layer_id, n_samples, arch)
#                 self.inspect_transformation_handles[layer_id] = layer.register_forward_pre_hook(hook)
            

#     def _get_train_transformation_hook(
#         self, transformation: nn.Module, layer_id: str
#     ) -> Callable:
#         def hook(module: nn.Module, x: Tuple[Tensor]) -> Tensor:
#             x_in, *_ = x  # tuple, alternatively use x_in = x[0]
#             x_in_new = transformation(x_in)
#             if layer_id not in self.disabled_ids:
#                 mse = nn.functional.mse_loss(x_in_new, x_in.detach(), reduction="mean")

#                 training_data = {
#                     "mse": mse,
#                 }

#                 self.training_data[layer_id] = training_data

#             return x_in_new

#         return hook
    

#     def _get_inference_transformation_hook(
#         self, transformation: nn.Module, n_samples: int = 1
#     ) -> Callable:
#         def hook(module: nn.Module, x: Tuple[Tensor]) -> Tensor:
#             x_in, *_ = x  # weird tuple, can use x_in = x[0]
#             if n_samples == 0:
#                 return x
#             elif n_samples == -1:
#                 x_in_new = transformation(x_in)
#                 return x_in_new
#             else:
#                 x_in_new = x_in.unsqueeze(1).repeat(1, n_samples, 1, 1, 1).flatten(0, 1)
#                 x_in_new = transformation(x_in_new)
#                 return torch.cat([x_in, x_in_new], dim=0)

#         return hook
            
        
#     def _get_inspect_transformation_hook(
#             self, 
#             transformation: nn.Module, 
#             layer_id: str, 
#             n_samples: int,
#             arch: str = 'ae',
#         ) -> Callable:
        
#         @torch.no_grad()
#         def hook(module: nn.Module, x: Tuple[Tensor]) -> Tensor:
#             x_in, *_ = x  # weird tuple, can use x_in = x[0]
#             if n_samples == 0:
#                 return x
#             elif n_samples == -1:
#                 mu, log_var, x_in_new = transformation(x_in)
#             else:
#                 x_in_new = x_in.unsqueeze(1).repeat(1, n_samples, 1, 1, 1).flatten(0, 1)
#                 if arch == 'ae':
#                     x_in_new = transformation(x_in_new)
#                 elif arch == 'res_ae':
#                     x_in_new, prior, residual = transformation(x_in_new)
#                 x_in_new = torch.cat([x_in, x_in_new], dim=0)
                
#             if layer_id not in self.disabled_ids:
#                 training_data = {
#                     'input'  : x_in_new[ :1],
#                     'recon'  : x_in_new[1: ],
#                 }
                
#                 if arch == 'res_ae':
#                     training_data['prior'] = prior
#                     training_data['residual'] = residual
                
#                 self.inspect_data[layer_id] = training_data
            
#             return x_in_new
        
#         return hook


#     def remove_train_transformation_hook(self, layer_id: str) -> None:
#         self.train_transformation_handles[layer_id].remove()

#     def remove_transformation_hook(self, layer_id: str) -> None:
#         self.transformation_handles[layer_id].remove()
        
#     def remove_inspect_transformation_hook(self, layer_id: str) -> None:
#         self.inspect_transformation_handles[layer_id].remove()

#     def remove_all_hooks(self):
#         if hasattr(self, "train_transformation_handles"):
#             for handle in self.train_transformation_handles:
#                 self.train_transformation_handles[handle].remove()
#             self.train_transformation_handles = {}

#         if hasattr(self, "transformation_handles"):
#             for handle in self.transformation_handles:
#                 self.transformation_handles[handle].remove()
#             self.transformation_handles = {}
            
#         if hasattr(self, 'inspect_transformation_handles'):
#             for handle in self.inspect_transformation_handles:
#                 self.inspect_transformation_handles[handle].remove()
#             self.inspect_transformation_handles = {}
        

#     def freeze_seg_model(self):
#         self.seg_model.eval()
#         for param in self.seg_model.parameters():
#             param.requires_grad = False

#     def set_number_of_samples_to(self, n_samples: int):
#         self.n_samples = n_samples

#     def disable(self, layer_ids: list) -> None:
#         for layer_id in layer_ids:
#             self.transformations[layer_id].turn_off()

#     def enable(self, layer_ids: list) -> None:
#         for layer_id in layer_ids:
#             self.transformations[layer_id].turn_on()

#     def forward(self, x: Tensor):
#         return self.seg_model(x)

    
    
    
# class ModelAdapter(nn.Module):
#     """Wrapper class for segmentation models and feature transformations.

#     Wraps (a copy of) the segmentation model and attaches feature
#     trasformations to it via hooks (at potentially various positions
#     simultaneously). Additionally, it provides control utilities for the
#     hooks as well as different types for inference and training.
#     """

#     def __init__(
#         self,
#         seg_model: nn.Module,
#         transformations: nn.ModuleDict,
#         disabled_ids: list = [],
#         copy: bool = True,
#     ):
#         super().__init__()
#         self.seg_model = deepcopy(seg_model) if copy else seg_model

#         self.transformations = transformations
#         self.disabled_ids = disabled_ids
#         self.transformation_handles = {}
#         self.train_transformation_handles = {}
#         self.inspect_transformation_handles = {}
#         self.training_data = {}
#         self.inspect_data = {}

#     def hook_train_transformations(
#         self, 
#         transformations: Dict[str, nn.Module]
#     ) -> None:
#         for layer_id in transformations:
#             layer = self.seg_model.get_submodule(layer_id)
#             hook = self._get_train_transformation_hook(
#                 transformations[layer_id], layer_id
#             )
#             self.train_transformation_handles[
#                 layer_id
#             ] = layer.register_forward_pre_hook(hook)

#     def hook_inference_transformations(
#         self, transformations: Dict[str, nn.Module], 
#         n_samples: int
#     ) -> None:
#         for layer_id in transformations:
#             layer = self.seg_model.get_submodule(layer_id)
#             hook = self._get_inference_transformation_hook(transformations[layer_id], n_samples)
#             self.transformation_handles[layer_id] = layer.register_forward_pre_hook(
#                 hook
#             )
            
#     def hook_inspect_transformation(
#         self, 
#         transformations: Dict[str, nn.Module], 
#     ) -> None:
#         for layer_id in transformations:
#             if layer_id not in self.disabled_ids:
#                 layer = self.seg_model.get_submodule(layer_id)
#                 hook  = self._get_inspect_transformation_hook(transformations[layer_id], layer_id)
#                 self.inspect_transformation_handles[layer_id] = layer.register_forward_pre_hook(hook)
            

# #     def _get_train_transformation_hook(
# #         self, transformation: nn.Module, layer_id: str
# #     ) -> Callable:
# #         def hook(module: nn.Module, x: Tuple[Tensor]) -> Tensor:
# #             x_in, *_ = x  # tuple, alternatively use x_in = x[0]
# #             x_orig = x_in[:1]
# #             #x_views = x_in[1:]
# #             x_in_denoised = transformation(x_in)
            
# #             if layer_id not in self.disabled_ids:
# #                 mse = nn.functional.mse_loss(x_in_denoised, x_orig.detach(), reduction="mean")

# #                 training_data = {
# #                     "mse": mse,
# #                 }

# #                 self.training_data[layer_id] = training_data

# #             #return torch.cat([x_orig, x_in_denoised], dim=0)
# #             return x_in_denoised
            
# #         return hook
    
    
#     def _get_train_transformation_hook(
#         self,
#         transformation: nn.Module,
#         layer_id: str
#     ) -> Callable:
#         def hook(module: nn.Module, x: Tuple[Tensor]) -> Tensor:
#             x_in, *_ = x  # tuple, alternatively use x_in = x[0]
#             #print(x_in.shape, x_in.shape[0] // 2)
#             batch_size = x_in.shape[0] // 2
#             x_orig, _  = torch.split(x_in, batch_size)
#             x_in_denoised = transformation(x_in)
            
#             if layer_id not in self.disabled_ids:
#                 mse = nn.functional.mse_loss(
#                     x_in_denoised, 
#                     x_orig.repeat(2,1,1,1).detach(),
#                     reduction="mean"
#                 )

#                 training_data = {
#                     "mse": mse,
#                 }

#                 self.training_data[layer_id] = training_data

#             return torch.cat([x_orig, x_in_denoised[batch_size:]], dim=0)
#             #return x_in_denoised
            
#         return hook
    

#     def _get_inference_transformation_hook(
#         self, transformation: nn.Module, n_samples: int = 1
#     ) -> Callable:
#         def hook(module: nn.Module, x: Tuple[Tensor]) -> Tensor:
#             x_in, *_ = x  # weird tuple, can use x_in = x[0]
#             if n_samples == 0:
#                 return x
#             elif n_samples == -1:
#                 x_in_new = transformation(x_in)
#                 return x_in_new
#             else:
#                 x_in_new = x_in.unsqueeze(1).repeat(1, n_samples, 1, 1, 1).flatten(0, 1)
#                 x_in_new = transformation(x_in_new)
#                 return torch.cat([x_in, x_in_new], dim=0)

#         return hook
            
        
# #     def _get_inspect_transformation_hook(
# #             self, 
# #             transformation: nn.Module, 
# #             layer_id: str, 
# #             n_samples: int,
# #             arch: str = 'ae',
# #         ) -> Callable:
        
# #         @torch.no_grad()
# #         def hook(module: nn.Module, x: Tuple[Tensor]) -> Tensor:
# #             x_in, *_ = x  # weird tuple, can use x_in = x[0]
# #             if n_samples == 0:
# #                 return x
# #             elif n_samples == -1:
# #                 mu, log_var, x_in_new = transformation(x_in)
# #             else:
# #                 x_in_new = x_in.unsqueeze(1).repeat(1, n_samples, 1, 1, 1).flatten(0, 1)
# #                 if arch == 'ae':
# #                     x_in_new = transformation(x_in_new)
# #                 elif arch == 'res_ae':
# #                     x_in_new, prior, residual = transformation(x_in_new)
# #                 x_in_new = torch.cat([x_in, x_in_new], dim=0)
                
# #             if layer_id not in self.disabled_ids:
# #                 training_data = {
# #                     'input'  : x_in_new[ :1],
# #                     'recon'  : x_in_new[1: ],
# #                 }
                
# #                 if arch == 'res_ae':
# #                     training_data['prior'] = prior
# #                     training_data['residual'] = residual
                
# #                 self.inspect_data[layer_id] = training_data
            
# #             return x_in_new
        
# #         return hook
    
    
#     def _get_inspect_transformation_hook(
#             self, 
#             transformation: nn.Module, 
#             layer_id: str, 
#         ) -> Callable:
        
#         @torch.no_grad()
#         def hook(module: nn.Module, x: Tuple[Tensor]) -> Tensor:
#             x_in, *_ = x  # weird tuple, can use x_in = x[0]
# #             #print(x_in.shape, x_in.shape[0] // 2)
# #             batch_size = x_in.shape[0] // 2
# #             x_orig, x_views  = torch.split(x_in, batch_size)
# #             x_in_denoised = transformation(x_views)
            
#             x_orig = x_in[:1]
#             x_in_denoised = transformation(x_in)
#             residuals     = x_in_denoised - x_in
                
#             if layer_id not in self.disabled_ids:
#                 data = {
#                     'input'     : x_in,
#                     'denoised'  : x_in_denoised,
#                     'residuals' : residuals
#                 }
                
#                 self.inspect_data[layer_id] = data
            
#             return torch.cat([x_orig, x_in_denoised[1:]], dim=0)
        
#         return hook
   
    

#     def remove_train_transformation_hook(self, layer_id: str) -> None:
#         self.train_transformation_handles[layer_id].remove()

#     def remove_transformation_hook(self, layer_id: str) -> None:
#         self.transformation_handles[layer_id].remove()
        
#     def remove_inspect_transformation_hook(self, layer_id: str) -> None:
#         self.inspect_transformation_handles[layer_id].remove()

#     def remove_all_hooks(self):
#         if hasattr(self, "train_transformation_handles"):
#             for handle in self.train_transformation_handles:
#                 self.train_transformation_handles[handle].remove()
#             self.train_transformation_handles = {}

#         if hasattr(self, "transformation_handles"):
#             for handle in self.transformation_handles:
#                 self.transformation_handles[handle].remove()
#             self.transformation_handles = {}
            
#         if hasattr(self, 'inspect_transformation_handles'):
#             for handle in self.inspect_transformation_handles:
#                 self.inspect_transformation_handles[handle].remove()
#             self.inspect_transformation_handles = {}
        

#     def freeze_seg_model(self):
#         self.seg_model.eval()
#         for param in self.seg_model.parameters():
#             param.requires_grad = False

#     def set_number_of_samples_to(self, n_samples: int):
#         self.n_samples = n_samples

#     def disable(self, layer_ids: list) -> None:
#         for layer_id in layer_ids:
#             self.transformations[layer_id].turn_off()

#     def enable(self, layer_ids: list) -> None:
#         for layer_id in layer_ids:
#             self.transformations[layer_id].turn_on()

#     def forward(self, x: Tensor):
#         return self.seg_model(x)

