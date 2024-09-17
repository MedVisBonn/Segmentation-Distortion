from torch import nn, flatten, sigmoid


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
        return sigmoid(output)