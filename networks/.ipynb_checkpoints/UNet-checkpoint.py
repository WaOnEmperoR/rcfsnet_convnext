import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms.functional as transforms

class CNNBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0):
        super(CNNBlock, self).__init__()

        self.seq_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.seq_block(x)
        return x

class CNNBlocks(nn.Module):
    """
    Parameters:
    n_conv (int): creates a block of n_conv convolutions
    in_channels (int): number of in_channels of the first block's convolution
    out_channels (int): number of out_channels of the first block's convolution
    expand (bool) : if True after the first convolution of a blocl the number of channels doubles
    """
    def __init__(self,
                 n_conv,
                 in_channels,
                 out_channels,
                 padding):
        super(CNNBlocks, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(n_conv):

            self.layers.append(CNNBlock(in_channels, out_channels, padding=padding))
            # after each convolution we set (next) in_channel to (previous) out_channels
            in_channels = out_channels

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Encoder(nn.Module):
    """
    Parameters:
    in_channels (int): number of in_channels of the first CNNBlocks
    out_channels (int): number of out_channels of the first CNNBlocks
    padding (int): padding applied in each convolution
    downhill (int): number times a CNNBlocks + MaxPool2D it's applied.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 padding,
                 downhill=4):
        super(Encoder, self).__init__()
        self.enc_layers = nn.ModuleList()

        for _ in range(downhill):
            self.enc_layers += [
                    CNNBlocks(n_conv=2, in_channels=in_channels, out_channels=out_channels, padding=padding),
                    nn.MaxPool2d(2, 2)
                ]

            in_channels = out_channels
            out_channels *= 2
        # doubling the dept of the last CNN block
        self.enc_layers.append(CNNBlocks(n_conv=2, in_channels=in_channels,
                                         out_channels=out_channels, padding=padding))

    def forward(self, x):
        route_connection = []
        for layer in self.enc_layers:
            if isinstance(layer, CNNBlocks):
                x = layer(x)
                route_connection.append(x)
            else:
                x = layer(x)
#             print("encoder: " + str(x.shape))
        return x, route_connection

class Decoder(nn.Module):
    """
    Parameters:
    in_channels (int): number of in_channels of the first ConvTranspose2d
    out_channels (int): number of out_channels of the first ConvTranspose2d
    padding (int): padding applied in each convolution
    uphill (int): number times a ConvTranspose2d + CNNBlocks it's applied.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 exit_channels,
                 padding,
                 uphill=4):
        super(Decoder, self).__init__()
        self.exit_channels = exit_channels
        self.layers = nn.ModuleList()

        for i in range(uphill):

            self.layers += [
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                CNNBlocks(n_conv=2, in_channels=in_channels,
                          out_channels=out_channels, padding=padding),
            ]
            in_channels //= 2
            out_channels //= 2

        # cannot be a CNNBlock because it has ReLU incorpored
        # cannot append nn.Sigmoid here because you should be later using
        # BCELoss () which will trigger the amp error "are unsafe to autocast".
        self.layers.append(
            nn.Conv2d(in_channels, exit_channels, kernel_size=1, padding=0),
        )
        

    def forward(self, x, routes_connection):
        # pop the last element of the list since
        # it's not used for concatenation
        routes_connection.pop(-1)
        for layer in self.layers:
            if isinstance(layer, CNNBlocks):
                # center_cropping the route tensor to make width and height match
                routes_connection[-1] = transforms.center_crop(routes_connection[-1], x.shape[2])
                # concatenating tensors channel-wise
                x = torch.cat([x, routes_connection.pop(-1)], dim=1)
                x = layer(x)
            else:
                x = layer(x)
#             print("decoder: " + str(x.shape))
        return x

class UNet(nn.Module):
    def __init__(self,
                 in_channels=3,
                 first_out_channels=64,
                 exit_channels=1,
                 downhill=4,
                 padding=1
                 ):
        super(UNet, self).__init__()
        super().__init__()
        self.encoder = Encoder(in_channels, first_out_channels, padding=padding, downhill=downhill)
        self.decoder = Decoder(first_out_channels*(2**downhill), first_out_channels*(2**(downhill-1)),
                               exit_channels, padding=padding, uphill=downhill)

    def forward(self, inputs):
        enc_out, routes = self.encoder(inputs)
        out = self.decoder(enc_out, routes)
        
        out = F.sigmoid(out) 
        
        return out