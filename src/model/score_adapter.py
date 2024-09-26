from typing import (
    List, 
    Callable, 
    Tuple
)
from torch import (
    nn, 
    no_grad, 
    Tensor, 
    nan_to_num, 
    argmax, 
    stack, 
    flatten, 
    sigmoid
)
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchmetrics.functional.classification import dice
from tqdm import tqdm
from copy import deepcopy

from utils import find_shapes_for_swivels




def fit_score_prediction_modification(
    wrapper: nn.Module,
    dataloader: DataLoader,
    score: Callable = dice,
    loss_fn: Callable = nn.MSELoss(reduction='none'),
    lr: float = 1e-6,
    epochs: int = 2500,
    device: str = 'cuda:0'
):
    wrapper.freeze_normalization_layers()
    wrapper.adapters.train()

    optimizer    = Adam(wrapper.parameters(), lr=lr)
    loss_cache   = []
    progress_bar = tqdm(range(epochs), total=epochs, position=0, leave=True)

    for i in progress_bar:
        
        batch  = next(dataloader)
        input  = batch['data'].to(device)
        target = batch['target'].long().to(device)
        target[target == -1] = 0
        logits = wrapper(input, target)
        prediction = wrapper.prediction
        
        preds = argmax(logits, dim=1, keepdim=True)
        DSCs  = stack([
                score(
                    s.flatten(), 
                    t.flatten(),
                    num_classes=4,
                    zero_division=1,
                    average='none'
                ) for s,t in zip(preds, target)
            ]).detach()
        
        DSCs.nan_to_num_(0)
        loss = loss_fn(prediction, DSCs)
        loss_aggregated = loss.sum(1).mean()
        loss_aggregated.backward()
        optimizer.step()

        loss_cache.append(loss.data.detach().cpu())
        epoch_summary = f"Loss: {loss_aggregated.item():.4f} | Progress"
        progress_bar.set_description("".join(epoch_summary))

    wrapper.eval()

    return loss_cache



def get_score_prediction_modification(
    swivels: List[str],
    unet: nn.Module,
    adapter_output_dim: int = 16,
    attach_prediction_head: bool = True,
    prediction_head_output_dim: int = 4,
    transform: bool = False,
    lr: float = 1e-3,
    return_state_dict: bool = False,
    device: str = 'cuda:0'
):
    
    # init prediction head
    if attach_prediction_head:
        prediction_head = PredictionHead(
            input_dim=len(swivels) * adapter_output_dim,
            output_dim=prediction_head_output_dim
        )
    else:
        prediction_head = None

    output_shapes = find_shapes_for_swivels(unet, swivels)
    # init detectors with corresponding predictors
    score_detectors = [
        ScorePredictionAdapter(
            swivel=swivel,
            predictor=ScorePredictor(
                input_size=output_shapes[swivel],
                output_dim=adapter_output_dim
            ),
            lr = lr,
            transform = transform,
        ) for swivel in swivels
    ]
    # wrap the model with the detectors
    wrapper = ScorePredictionWrapper(
        model=unet,
        prediction_head=prediction_head,
        adapters=nn.ModuleList(score_detectors),
        copy=True
    )
    wrapper.freeze_model()
    wrapper.to(device)

    if return_state_dict:
        raise NotImplementedError
    
    return wrapper




class PredictionHead(nn.Module):
    def __init__(
            self, 
            input_dim, 
            output_dim
        ):
        super(PredictionHead, self).__init__()
        hidden_dim = input_dim // 4
        # Define the first fully connected layer
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # Define Layer Normalization for the first layer
        self.ln1 = nn.LayerNorm(hidden_dim)
        # Leaky ReLU activation function for the first layer
        self.leaky_relu1 = nn.LeakyReLU(negative_slope=0.01)
        
        # Define the second fully connected layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Define Layer Normalization for the second layer
        self.ln2 = nn.LayerNorm(hidden_dim)
        # Leaky ReLU activation function for the second layer
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.01)
        
        # Define the third fully connected layer (output layer)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Forward pass through the first layer, normalization, and activation
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.leaky_relu1(x)
        
        # Forward pass through the second layer, normalization, and activation
        x = self.fc2(x)
        x = self.ln2(x)
        x = self.leaky_relu2(x)
        
        # Output layer (no activation here if it's a regression task or depending on the task)
        x = self.fc3(x)
        
        return x



class ScorePredictor(nn.Module):
    """
    A simple model to predict a score based on a DNN activation.

    Args:
    - input_size (list): The size of the input tensor
    - hidden_dim (int): The size of the hidden layer
    - output_dim (int): The size of the output layer
    - dropout_prob (float): The probability of dropout

    Returns:
    - output (torch.Tensor): The output of the model
    """
    def __init__(
        self, 
        input_size: list, 
        hidden_dim: int = 128, 
        output_dim: int = 1, 
        dropout_prob=0.3
    ):
        super(ScorePredictor, self).__init__()
        
        # 1x1 Conv Layer to reduce the number of input channels by a factor of 8
        self.conv1x1 = nn.Conv2d(
            in_channels=input_size[1], 
            out_channels=input_size[1] // 8, 
            kernel_size=1
        )

        self.conv2x2_1 = nn.Conv2d(
            in_channels=input_size[1] // 8,
            out_channels=input_size[1] // 8,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.relu = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(input_size[1] // 8)

        self.conv2x2_2 = nn.Conv2d(
            in_channels=input_size[1] // 8,
            out_channels=input_size[1] // 8,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.relu = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(input_size[1] // 8)
        
        # Activation, Norm, Dropout
        self.activation1 = nn.LeakyReLU()  # You can also use other activations like Swish or LeakyReLU
        self.norm1 = nn.BatchNorm2d(input_size[1] // 8)
        self.dropout1 = nn.Dropout(dropout_prob)
        
        # Compute flattened size after convolution for the linear layers
        conv_output_size = input_size[1] * input_size[2]*input_size[3] // 8
        
        # Fully connected layer to reduce the output of the conv to hidden_dim
        self.fc1 = nn.Linear(conv_output_size, hidden_dim)
        self.activation2 = nn.LeakyReLU()
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Final fully connected layer to output_dim
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Pass through 1x1 convolution, activation, normalization, and dropout
        x = self.conv1x1(x)

        x = self.relu(x)
        x = self.bn1(x)
        x = self.conv2x2_1(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.conv2x2_2(x)

        x = self.activation1(x)
        x = self.norm1(x)
        x = self.dropout1(x)
        
        # Flatten the output from the conv layer
        x = flatten(x, start_dim=1)
        
        # Fully connected layer to hidden_dim
        x = self.fc1(x)
        x = self.activation2(x)
        x = self.norm2(x)
        
        # Final output layer
        output = self.fc2(x)
        return output #sigmoid(output)



class ScorePredictionAdapter(nn.Module):
    def __init__(
        self,
        swivel: str,
        predictor: nn.Module,
        train_lr: float = 1e-3,
        lr: float = 1e-3,
        transform: bool = False,
        device: str = 'cuda:0'
    ):
        super().__init__()
        self.swivel = swivel
        self.predictor = predictor
        self.train_lr = train_lr
        self.lr = lr
        self.transform = transform
        self.device = device
        self.active = True

        self.to(device)
        self.score_collection = []


    @no_grad()
    def _collect(
        self, 
        x: Tensor
    ) -> None:
        # reduces dimensionality as per self._pool, moves to cpu and stores
        x = self.predictor(x.detach()).cpu()
        self.score_collection.append(x)


    def on(self):
        self.active = True


    def off(self):
        self.active = False


    def forward(
        self,
        x: Tensor
    ) -> Tensor:
        # this adapter only operates of turned on
        if self.active:
            if self.transform:
                assert not self.training
                x = x.clone().detach().requires_grad_(True)
                self.score = self.predictor(x).sum()
                self.score.backward()
                x.grad.data = nan_to_num(x.grad.data, nan=0.0)
                x.data.sub_(self.lr * x.grad.data)
                x.grad.data.zero_()
                x.requires_grad = False

            else:
                self.score = self.predictor(x)

        else:
            pass
        return x



class ScorePredictionWrapper(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        prediction_head: nn.Module,
        adapters: nn.ModuleList,
        copy: bool = True,
    ):
        super().__init__()
        self.model           = deepcopy(model) if copy else model
        self.prediction_head = prediction_head
        self.adapters        = adapters
        self.adapter_handles = {}
        self.transform       = False
        self.model.eval()

        self.hook_adapters()


    def hook_adapters(
        self,
    ) -> None:
        assert self.adapter_handles == {}, "Adapters already hooked"
        for adapter in self.adapters:
            swivel = adapter.swivel
            layer  = self.model.get_submodule(swivel)
            hook   = self._get_hook(adapter)
            self.adapter_handles[
                swivel
            ] = layer.register_forward_pre_hook(hook)


    def _get_hook(
        self,
        adapter: nn.Module
    ) -> Callable:
        def hook_fn(
            module: nn.Module, 
            x: Tuple[Tensor]
        ) -> Tensor:
            # x, *_ = x # tuple, alternatively use x_in = x[0]
            # x = adapter(x)
            return adapter(x[0])
        
        return hook_fn
    

    def set_transform(self, transform: bool):
        self.transform = transform
        for adapter in self.adapters:
            adapter.transform = transform


    def set_train_lr(self, train_lr):
        for adapter in self.adapters:
            adapter.train_lr = train_lr


    def set_train_lr(self, inference_lr):
        for adapter in self.adapters:
            adapter.inference_lr = inference_lr


    def turn_off_all_adapters(self):
        for adapter in self.adapters:
            adapter.off()

    
    def freeze_model(self):
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False


    def freeze_normalization_layers(self):
        for name, module in self.model.named_modules():
            if 'bn' in name:
                module.eval()
            


    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
    
        logits = self.model(x).detach()
        
        self.output_per_adapter = stack([
            adapter.score for adapter in self.adapters
        ], dim=1)

        if self.prediction_head is not None:
            self.prediction = self.prediction_head(self.output_per_adapter.flatten(1))

        return logits