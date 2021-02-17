from torch import nn
from torch.optim import Adam

from mask_generators import ImageMaskGenerator
from nn_utils import MemoryLayer, SkipConnection, ResBlock3d
from prob_utils import normal_parse_params, GaussianLoss


class CBatchNorm2d(nn.Module):
    def __init__(self, dim, n_cond=0):
        super().__init__()

        self.bn = nn.BatchNorm2d(dim)
        self.gain = nn.Linear(n_cond, dim)
        self.bias = nn.Linear(n_cond, dim)

    def forward(self, x,  y):
        gain = (1 + self.gain(y)).view(y.size(0), -1, 1, 1)
        bias = self.bias(y).view(y.size(0), -1, 1, 1)

        return self.bn(x) * gain + bias

class CResBlock(nn.Module):
    """
    Usual full pre-activation ResNet bottleneck block.
    For more information see
    He, K., Zhang, X., Ren, S., & Sun, J. (2016, October).
    Identity mappings in deep residual networks.
    European Conference on Computer Vision (pp. 630-645).
    Springer, Cham.
    ArXiv link: https://arxiv.org/abs/1603.05027
    """

    def __init__(self, outer_dim, inner_dim, n_cond=0):
        super().__init__()
        
        self.bn1 = CBatchNorm2d(outer_dim, n_cond=n_cond)
        self.bn2 = CBatchNorm2d(inner_dim, n_cond=n_cond)
        self.bn3 = CBatchNorm2d(inner_dim, n_cond=n_cond)

        self.lrelu = nn.LeakyReLU()
        self.conv1 = nn.Conv2d(outer_dim, inner_dim, 1)
        self.conv2 = nn.Conv2d(inner_dim, inner_dim, 3, 1, 1)
        self.conv3 = nn.Conv2d(inner_dim, outer_dim, 1)


    def forward(self, input, y):
        
        out = self.bn1(input, y)
        out = self.lrelu(out)
        out = self.conv1(out)

        out = self.bn2(out, y)
        out = self.lrelu(out)
        out = self.conv2(out)

        out = self.bn3(out, y)
        out = self.lrelu(out)
        out = self.conv3(out)

        return input + out


class ResBlock(nn.Module):
    """
    Usual full pre-activation ResNet bottleneck block.
    For more information see
    He, K., Zhang, X., Ren, S., & Sun, J. (2016, October).
    Identity mappings in deep residual networks.
    European Conference on Computer Vision (pp. 630-645).
    Springer, Cham.
    ArXiv link: https://arxiv.org/abs/1603.05027
    """

    def __init__(self, outer_dim, inner_dim, cond=None):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(outer_dim),
            nn.LeakyReLU(),
            nn.Conv2d(outer_dim, inner_dim, 1),
            nn.BatchNorm2d(inner_dim),
            nn.LeakyReLU(),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.BatchNorm2d(inner_dim),
            nn.LeakyReLU(),
            nn.Conv2d(inner_dim, outer_dim, 1),
        )


    def forward(self, input):
        # print(input.shape, "res")
        return input + self.net(input)



class SkipConnection(nn.Module):
    """
    Skip-connection over the sequence of layers in the constructor.
    The module passes input data sequentially through these layers
    and then adds original data to the result.
    """
    def __init__(self, *args):
        super().__init__()
        self.inner_net = nn.Sequential(*args)

    def forward(self, input):
        return input + self.inner_net(input)


class UpSampleResBlock(nn.Module):
    """
    Usual full pre-activation ResNet bottleneck block.
    For more information see
    He, K., Zhang, X., Ren, S., & Sun, J. (2016, October).
    Identity mappings in deep residual networks.
    European Conference on Computer Vision (pp. 630-645).
    Springer, Cham.
    ArXiv link: https://arxiv.org/abs/1603.05027
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_dim, out_dim, 1),

        )

        self.net2 = nn.Sequential(
            nn.BatchNorm2d(in_dim),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_dim, out_dim, 3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
            nn.Conv2d(out_dim, out_dim, 3, padding=1)
            
        )

    def forward(self, input):
        # print(input.shape, "res")
        return self.net1(input) + self.net2(input)

class DownSampleResBlock(nn.Module):
    """
    Usual full pre-activation ResNet bottleneck block.
    For more information see
    He, K., Zhang, X., Ren, S., & Sun, J. (2016, October).
    Identity mappings in deep residual networks.
    European Conference on Computer Vision (pp. 630-645).
    Springer, Cham.
    ArXiv link: https://arxiv.org/abs/1603.05027
    """

    def __init__(self, in_dim, out_dim, batchnorm):
        super().__init__()
        self.net1 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 1),
            nn.AvgPool2d(2, 2),
        )

        net_list = [
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(in_dim, out_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_dim, out_dim, 3, padding=1)
        ]

        if batchnorm:
            net_list.insert(0, nn.BatchNorm2d(in_dim))
            net_list.insert(4, nn.BatchNorm2d(out_dim))


        self.net2 = nn.Sequential(*net_list)

    def forward(self, input):
        # print(input.shape, "res")
        return self.net1(input) + self.net2(input)

# sampler from the model generative distribution
# here we return mean of the Gaussian to avoid white noise
def sampler(params):
    return normal_parse_params(params).mean


def optimizer(parameters, lr=2e-4):
    return Adam(parameters, lr=lr)

# reconstruction_log_prob = GaussianLoss()
# reconstruction_log_prob = L1L


mask_generator = ImageMaskGenerator()

# improve train computational stability by dividing the loss
# by this scale factor right before backpropagation
vlb_scale_factor = 192 ** 2

# def MLPBlock(dim):
#     return SkipConnection(
#         nn.BatchNorm2d(dim),
#         nn.LeakyReLU(),
#         nn.Conv2d(dim, dim, 1)
    # )
def MLPBlock(dim):
    return SkipConnection(
        nn.BatchNorm2d(dim),
        nn.LeakyReLU(),
        nn.Conv2d(dim, dim, 1)
    )

def ds_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1),
        nn.LeakyReLU()
    )





def get_networks(in_channels=1, z_dim=256):

    proposal_network = nn.Sequential(
        nn.Conv2d(2, 8, 1),
        # ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
        # nn.AvgPool2d(2, 2),
        # ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
        # nn.AvgPool2d(2, 2),
        ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
        nn.AvgPool2d(2, 2),
        ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
        nn.AvgPool2d(2, 2), nn.Conv2d(8, 16, 1),
        ResBlock(16, 8), ResBlock(16, 8), ResBlock(16, 8), ResBlock(16, 8),
        nn.AvgPool2d(2, 2), nn.Conv2d(16, 32, 1),
        ResBlock(32, 16), ResBlock(32, 16), ResBlock(32, 16), ResBlock(32, 16),
        nn.Conv2d(32, 64, 1),
        ResBlock(64, 32), ResBlock(64, 32), ResBlock(64, 32), ResBlock(64, 32),
        nn.Conv2d(64, 128, 1),
        ResBlock(128, 64), ResBlock(128, 64),
        ResBlock(128, 64), ResBlock(128, 64),
        nn.Conv2d(128, 256, 1),
        ResBlock(256, 128), ResBlock(256, 128),
        ResBlock(256, 128), ResBlock(256, 128),
        nn.Conv2d(256, z_dim, 1),
        # MLPBlock(512), MLPBlock(512), MLPBlock(512), MLPBlock(512),
    )

    prior_network = nn.Sequential(
        MemoryLayer('#0'),
        nn.Conv2d(2, 8, 1),
        # ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
        # MemoryLayer('#1'),
        # nn.AvgPool2d(2, 2),
        # ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
        # MemoryLayer('#2'),
        # nn.AvgPool2d(2, 2),
        ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
        MemoryLayer('#3'),
        nn.AvgPool2d(2, 2),
        ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
        MemoryLayer('#4'),
        nn.AvgPool2d(2, 2), nn.Conv2d(8, 16, 1),
        ResBlock(16, 8), ResBlock(16, 8), ResBlock(16, 8), ResBlock(16, 8),
        MemoryLayer('#5'),
        nn.AvgPool2d(2, 2), nn.Conv2d(16, 32, 1),
        ResBlock(32, 16), ResBlock(32, 16), ResBlock(32, 16), ResBlock(32, 16),
        MemoryLayer('#6'),
        nn.Conv2d(32, 64, 1),
        ResBlock(64, 32), ResBlock(64, 32), ResBlock(64, 32), ResBlock(64, 32),
        MemoryLayer('#7'),
        nn.Conv2d(64, 128, 1),
        ResBlock(128, 64), ResBlock(128, 64),
        ResBlock(128, 64), ResBlock(128, 64),
        MemoryLayer('#8'),
        nn.Conv2d(128, 256, 1),
        ResBlock(256, 128), ResBlock(256, 128),
        ResBlock(256, 128), ResBlock(256, 128),
        MemoryLayer('#9'),
        nn.Conv2d(256, z_dim, 1),
        # MLPBlock(512), MLPBlock(512), MLPBlock(512), MLPBlock(512),
    )

    generative_network = nn.Sequential(
        nn.Conv2d(z_dim // 2, 256, 1),
        MLPBlock(256), MLPBlock(256), MLPBlock(256), MLPBlock(256),
        nn.Conv2d(256, 128, 1),# nn.Upsample(scale_factor=3),
        MemoryLayer('#9', True), nn.Conv2d(384, 128, 1),
        ResBlock(128, 64), ResBlock(128, 64),
        ResBlock(128, 64), ResBlock(128, 64),
        nn.Conv2d(128, 64, 1),# nn.Upsample(scale_factor=2),
        MemoryLayer('#8', True), nn.Conv2d(192, 64, 1),
        ResBlock(64, 32), ResBlock(64, 32), ResBlock(64, 32), ResBlock(64, 32),
        nn.Conv2d(64, 32, 1),# nn.Upsample(scale_factor=2),
        MemoryLayer('#7', True), nn.Conv2d(96, 32, 1),
        ResBlock(32, 16), ResBlock(32, 16), ResBlock(32, 16), ResBlock(32, 16),
        nn.Conv2d(32, 16, 1),# nn.Upsample(scale_factor=2),
        MemoryLayer('#6', True), nn.Conv2d(48, 16, 1),
        ResBlock(16, 8), ResBlock(16, 8), ResBlock(16, 8), ResBlock(16, 8),
        nn.Conv2d(16, 8, 1), nn.Upsample(scale_factor=2),
        MemoryLayer('#5', True), nn.Conv2d(24, 8, 1),
        ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
        nn.Upsample(scale_factor=2),
        MemoryLayer('#4', True), nn.Conv2d(16, 8, 1),
        ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
        nn.Upsample(scale_factor=2),
        MemoryLayer('#3', True), nn.Conv2d(16, 8, 1),
        ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
        # nn.Upsample(scale_factor=2),
        # MemoryLayer('#2', True), nn.Conv2d(16, 8, 1),
        # ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
        # nn.Upsample(scale_factor=2),
        # MemoryLayer('#1', True), nn.Conv2d(16, 8, 1),
        # ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
        MemoryLayer('#0', True), nn.Conv2d(10, 8, 1),
        ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
        nn.Conv2d(8, 2, 1),
    )


    # proposal_network = nn.Sequential(
    #     nn.Conv2d(in_channels * 2, 8, 1),
    #     # ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
    #     # nn.AvgPool2d(2, 2),
    #     # ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
    #     # nn.AvgPool2d(2, 2),
    #     ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
    #     nn.AvgPool2d(2, 2),
    #     ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), # 96
    #     nn.AvgPool2d(2, 2), nn.Conv2d(8, 16, 1),
    #     ResBlock(16, 8), ResBlock(16, 8), ResBlock(16, 8), ResBlock(16, 8), # 48
    #     nn.AvgPool2d(2, 2), nn.Conv2d(16, 32, 1),
    #     ResBlock(32, 16), ResBlock(32, 16), ResBlock(32, 16), ResBlock(32, 16), # 24
    #     nn.Conv2d(32, min(z_dim, 64), 1),
    #     # ResBlock(64, 32), ResBlock(64, 32), ResBlock(64, 32), ResBlock(64, 32),
    #     ResBlock(min(z_dim, 64), max(z_dim // 2, 32)), 
    #     ResBlock(min(z_dim, 64), max(z_dim // 2, 32)), 
    #     ResBlock(min(z_dim, 64), max(z_dim // 2, 32)), 
    #     ResBlock(min(z_dim, 64), max(z_dim // 2, 32)), 
    #     nn.Conv2d(min(z_dim, 64), min(z_dim, 128), 1),
    #     # ResBlock(128, 64), ResBlock(128, 64), ResBlock(128, 64), ResBlock(128, 64),
    #     ResBlock(min(z_dim, 128), min(z_dim // 2, 64)), 
    #     ResBlock(min(z_dim, 128), min(z_dim // 2, 64)), 
    #     ResBlock(min(z_dim, 128), min(z_dim // 2, 64)), 
    #     ResBlock(min(z_dim, 128), min(z_dim // 2, 64)), 
    #     nn.Conv2d(min(z_dim, 128), min(z_dim, 256), 1),
    #     # ResBlock(256, 128), ResBlock(256, 128), ResBlock(256, 128), ResBlock(256, 128)
    #     ResBlock(min(z_dim, 256), min(z_dim // 2, 128)), 
    #     ResBlock(min(z_dim, 256), min(z_dim // 2, 128)), 
    #     ResBlock(min(z_dim, 256), min(z_dim // 2, 128)), 
    #     ResBlock(min(z_dim, 256), min(z_dim // 2, 128))
    #     # nn.Conv2d(256, 512, 1),
    #     # MLPBlock(512), MLPBlock(512), MLPBlock(512), MLPBlock(512),
    # )


    # prior_network = nn.Sequential(
    #     MemoryLayer('#0'),
    #     nn.Conv2d(in_channels * 2, 8, 1),
    #     # ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
    #     # MemoryLayer('#1'),
    #     # nn.AvgPool2d(2, 2),
    #     # ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
    #     # MemoryLayer('#2'),
    #     # nn.AvgPool2d(2, 2),
    #     ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
    #     MemoryLayer('#3'),
    #     nn.AvgPool2d(2, 2),
    #     ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
    #     MemoryLayer('#4'),
    #     nn.AvgPool2d(2, 2), nn.Conv2d(8, 16, 1),
    #     ResBlock(16, 8), ResBlock(16, 8), ResBlock(16, 8), ResBlock(16, 8),
    #     MemoryLayer('#5'),
    #     nn.AvgPool2d(2, 2), nn.Conv2d(16, 32, 1),
    #     ResBlock(32, 16), ResBlock(32, 16), ResBlock(32, 16), ResBlock(32, 16),
    #     MemoryLayer('#6'),
    #     nn.Conv2d(32, min(z_dim, 64), 1),
    #     # ResBlock(64, 32), ResBlock(64, 32), ResBlock(64, 32), ResBlock(64, 32),
    #     ResBlock(min(z_dim, 64), min(z_dim // 2, 32)), 
    #     ResBlock(min(z_dim, 64), min(z_dim // 2, 32)), 
    #     ResBlock(min(z_dim, 64), min(z_dim // 2, 32)), 
    #     ResBlock(min(z_dim, 64), min(z_dim // 2, 32)), 
    #     MemoryLayer('#7'),
    #     nn.Conv2d(min(z_dim, 64), min(z_dim, 128), 1),
    #     # nn.AvgPool2d(2, 2), nn.Conv2d(64, 128, 1),
    #     ResBlock(min(z_dim, 128), min(z_dim // 2, 64)), 
    #     ResBlock(min(z_dim, 128), min(z_dim // 2, 64)), 
    #     ResBlock(min(z_dim, 128), min(z_dim // 2, 64)), 
    #     ResBlock(min(z_dim, 128), min(z_dim // 2, 64)), 
    #     MemoryLayer('#8'),
    #     nn.Conv2d(min(z_dim, 128), min(z_dim, 256), 1),
    #     ResBlock(min(z_dim, 256), min(z_dim // 2, 128)), 
    #     ResBlock(min(z_dim, 256), min(z_dim // 2, 128)), 
    #     ResBlock(min(z_dim, 256), min(z_dim // 2, 128)), 
    #     ResBlock(min(z_dim, 256), min(z_dim // 2, 128)),
    #     MemoryLayer('#9'),
    #     nn.Conv2d(min(z_dim, 256), min(z_dim, 512), 1),
    #     # MLPBlock(512), MLPBlock(512), MLPBlock(512), MLPBlock(512),
    # )

    # # z_dim /= 2
    # generative_network = nn.Sequential(
    #     # nn.Conv2d(256, 256, 1),
    #     # MLPBlock(256), MLPBlock(256), MLPBlock(256), MLPBlock(256),
    #     nn.Conv2d(min(z_dim // 2, 256), min(z_dim // 2, 256), 1),
        
    #     # nn.Conv2d(256, 128, 1), nn.Upsample(scale_factor=3),
    #     # MemoryLayer('#9', True), nn.Conv2d(384, 128, 1),
    #     MemoryLayer('#9', True), 
    #     nn.Conv2d(min(z_dim, 256) + min(z_dim // 2, 256), min(z_dim // 2, 128), 1),

    #     ResBlock(min(z_dim // 2, 128), min(z_dim // 4, 64)), 
    #     ResBlock(min(z_dim // 2, 128), min(z_dim // 4, 64)), 
    #     ResBlock(min(z_dim // 2, 128), min(z_dim // 4, 64)), 
    #     ResBlock(min(z_dim // 2, 128), min(z_dim // 4, 64)), 
    #     nn.Conv2d(min(z_dim // 2, 128), min(z_dim // 2, 64), 1), #nn.Upsample(scale_factor=2),

    #     MemoryLayer('#8', True), 
    #     nn.Conv2d(min(z_dim, 128) + min(z_dim // 2, 64), min(z_dim // 2, 64), 1),
    #     ResBlock(min(z_dim // 2, 64), min(z_dim // 2, 32)), 
    #     ResBlock(min(z_dim // 2, 64), min(z_dim // 2, 32)), 
    #     ResBlock(min(z_dim // 2, 64), min(z_dim // 2, 32)), 
    #     ResBlock(min(z_dim // 2, 64), min(z_dim // 2, 32)),
    #     nn.Conv2d(min(z_dim // 2, 64), 32, 1), #nn.Upsample(scale_factor=2),

    #     MemoryLayer('#7', True), 
    #     nn.Conv2d(min(z_dim, 64) + 32, 32, 1),
    #     ResBlock(32, 16), ResBlock(32, 16), ResBlock(32, 16), ResBlock(32, 16),
    #     nn.Conv2d(32, 16, 1), #nn.Upsample(scale_factor=2),
        
    #     MemoryLayer('#6', True), nn.Conv2d(48, 16, 1),
    #     ResBlock(16, 8), ResBlock(16, 8), ResBlock(16, 8), ResBlock(16, 8),
    #     nn.Conv2d(16, 8, 1), nn.Upsample(scale_factor=2),
        
    #     MemoryLayer('#5', True), nn.Conv2d(24, 8, 1),
    #     ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
    #     nn.Upsample(scale_factor=2),
        
    #     MemoryLayer('#4', True), nn.Conv2d(16, 8, 1),
    #     ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
    #     nn.Upsample(scale_factor=2),
    #     MemoryLayer('#3', True), nn.Conv2d(16, 8, 1),
    #     ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
    #     # nn.Upsample(scale_factor=2),
    #     # MemoryLayer('#2', True), nn.Conv2d(16, 8, 1),
    #     # ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
    #     # nn.Upsample(scale_factor=2),
    #     # MemoryLayer('#1', True), nn.Conv2d(16, 8, 1),
    #     # ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
    #     MemoryLayer('#0', True), nn.Conv2d(8 + in_channels * 2, 8, 1),
    #     ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
    #     nn.Conv2d(8, in_channels * 2, 1),
    # )
    return proposal_network, prior_network, generative_network

def get_dense_networks(in_channels=1, z_dim=256):

    proposal_network = nn.Sequential(
        nn.Conv2d(in_channels, 8, 1),
        # ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
        # nn.AvgPool2d(2, 2),
        # ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
        # nn.AvgPool2d(2, 2),
        ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
        nn.AvgPool2d(2, 2),
        ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
        nn.AvgPool2d(2, 2), nn.Conv2d(8, 16, 1),
        ResBlock(16, 8), ResBlock(16, 8), ResBlock(16, 8), ResBlock(16, 8),
        nn.AvgPool2d(2, 2), nn.Conv2d(16, 32, 1),
        ResBlock(32, 16), ResBlock(32, 16), ResBlock(32, 16), ResBlock(32, 16),
        nn.AvgPool2d(2, 2), nn.Conv2d(32, 64, 1),
        ResBlock(64, 32), ResBlock(64, 32), ResBlock(64, 32), ResBlock(64, 32),
        nn.AvgPool2d(2, 2), nn.Conv2d(64, 128, 1),
        ResBlock(128, 64), ResBlock(128, 64),
        ResBlock(128, 64), ResBlock(128, 64),
        nn.AvgPool2d(2, 2), nn.Conv2d(128, 256, 1),
        ResBlock(256, 128), ResBlock(256, 128),
        ResBlock(256, 128), ResBlock(256, 128),
        nn.AvgPool2d((3, 3)), nn.Conv2d(256, 512, 1),
        MLPBlock(512), MLPBlock(512), MLPBlock(512), MLPBlock(512),
        nn.Conv2d(512, z_dim, 1)

    )

    prior_network = nn.Sequential(
        MemoryLayer('#0'),
        nn.Conv2d(in_channels, 8, 1),
        # ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
        # MemoryLayer('#1'),
        # nn.AvgPool2d(2, 2),
        # ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
        # MemoryLayer('#2'),
        # nn.AvgPool2d(2, 2),
        ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
        MemoryLayer('#3'),
        nn.AvgPool2d(2, 2),
        ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
        MemoryLayer('#4'),
        nn.AvgPool2d(2, 2), nn.Conv2d(8, 16, 1),
        ResBlock(16, 8), ResBlock(16, 8), ResBlock(16, 8), ResBlock(16, 8),
        MemoryLayer('#5'),
        nn.AvgPool2d(2, 2), nn.Conv2d(16, 32, 1),
        ResBlock(32, 16), ResBlock(32, 16), ResBlock(32, 16), ResBlock(32, 16),
        MemoryLayer('#6'),
        nn.AvgPool2d(2, 2), nn.Conv2d(32, 64, 1),
        ResBlock(64, 32), ResBlock(64, 32), ResBlock(64, 32), ResBlock(64, 32),
        MemoryLayer('#7'),
        nn.AvgPool2d(2, 2), nn.Conv2d(64, 128, 1),
        ResBlock(128, 64), ResBlock(128, 64),
        ResBlock(128, 64), ResBlock(128, 64),
        MemoryLayer('#8'),
        nn.AvgPool2d(2, 2), nn.Conv2d(128, 256, 1),
        ResBlock(256, 128), ResBlock(256, 128),
        ResBlock(256, 128), ResBlock(256, 128),
        MemoryLayer('#9'),
        nn.AvgPool2d((3, 3)), nn.Conv2d(256, 512, 1),
        MLPBlock(512), MLPBlock(512), MLPBlock(512), MLPBlock(512),
        nn.Conv2d(512, z_dim, 1)
    )

    generative_network = nn.Sequential(
        nn.Conv2d(z_dim // 2, 256, 1),
        MLPBlock(256), MLPBlock(256), MLPBlock(256), MLPBlock(256),
        nn.Conv2d(256, 128, 1), nn.Upsample(scale_factor=3),
        MemoryLayer('#9', True), nn.Conv2d(384, 128, 1),
        ResBlock(128, 64), ResBlock(128, 64),
        ResBlock(128, 64), ResBlock(128, 64),
        nn.Conv2d(128, 64, 1), nn.Upsample(scale_factor=2),
        MemoryLayer('#8', True), nn.Conv2d(192, 64, 1),
        ResBlock(64, 32), ResBlock(64, 32), ResBlock(64, 32), ResBlock(64, 32),
        nn.Conv2d(64, 32, 1), nn.Upsample(scale_factor=2),
        MemoryLayer('#7', True), nn.Conv2d(96, 32, 1),
        ResBlock(32, 16), ResBlock(32, 16), ResBlock(32, 16), ResBlock(32, 16),
        nn.Conv2d(32, 16, 1), nn.Upsample(scale_factor=2),
        MemoryLayer('#6', True), nn.Conv2d(48, 16, 1),
        ResBlock(16, 8), ResBlock(16, 8), ResBlock(16, 8), ResBlock(16, 8),
        nn.Conv2d(16, 8, 1), nn.Upsample(scale_factor=2),
        MemoryLayer('#5', True), nn.Conv2d(24, 8, 1),
        ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
        nn.Upsample(scale_factor=2),
        MemoryLayer('#4', True), nn.Conv2d(16, 8, 1),
        ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
        nn.Upsample(scale_factor=2),
        MemoryLayer('#3', True), nn.Conv2d(16, 8, 1),
        ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
        # nn.Upsample(scale_factor=2),
        # MemoryLayer('#2', True), nn.Conv2d(16, 8, 1),
        # ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
        # nn.Upsample(scale_factor=2),
        # MemoryLayer('#1', True), nn.Conv2d(16, 8, 1),
        # ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
        MemoryLayer('#0', True), nn.Conv2d(8 + in_channels, 8, 1),
        ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
        nn.Conv2d(8, 1, 1),
    )
    # proposal_network = nn.Sequential(
    #     nn.Conv2d(in_channels * 2, 8, 1),
    #     # ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
    #     # nn.AvgPool2d(2, 2),
    #     # ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
    #     # nn.AvgPool2d(2, 2),
    #     ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
    #     nn.AvgPool2d(2, 2),
    #     ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), # 96
    #     nn.AvgPool2d(2, 2), nn.Conv2d(8, 16, 1),
    #     ResBlock(16, 8), ResBlock(16, 8), ResBlock(16, 8), ResBlock(16, 8), # 48
    #     nn.AvgPool2d(2, 2), nn.Conv2d(16, 32, 1),
    #     ResBlock(32, 16), ResBlock(32, 16), ResBlock(32, 16), ResBlock(32, 16), # 24
    #     nn.Conv2d(32, min(z_dim, 64), 1),
    #     # ResBlock(64, 32), ResBlock(64, 32), ResBlock(64, 32), ResBlock(64, 32),
    #     ResBlock(min(z_dim, 64), max(z_dim // 2, 32)), 
    #     ResBlock(min(z_dim, 64), max(z_dim // 2, 32)), 
    #     ResBlock(min(z_dim, 64), max(z_dim // 2, 32)), 
    #     ResBlock(min(z_dim, 64), max(z_dim // 2, 32)), 
    #     nn.Conv2d(min(z_dim, 64), min(z_dim, 128), 1),
    #     # ResBlock(128, 64), ResBlock(128, 64), ResBlock(128, 64), ResBlock(128, 64),
    #     ResBlock(min(z_dim, 128), min(z_dim // 2, 64)), 
    #     ResBlock(min(z_dim, 128), min(z_dim // 2, 64)), 
    #     ResBlock(min(z_dim, 128), min(z_dim // 2, 64)), 
    #     ResBlock(min(z_dim, 128), min(z_dim // 2, 64)), 
    #     nn.Conv2d(min(z_dim, 128), min(z_dim, 256), 1),
    #     # ResBlock(256, 128), ResBlock(256, 128), ResBlock(256, 128), ResBlock(256, 128)
    #     ResBlock(min(z_dim, 256), min(z_dim // 2, 128)), 
    #     ResBlock(min(z_dim, 256), min(z_dim // 2, 128)), 
    #     ResBlock(min(z_dim, 256), min(z_dim // 2, 128)), 
    #     ResBlock(min(z_dim, 256), min(z_dim // 2, 128))
    #     # nn.Conv2d(256, 512, 1),
    #     # MLPBlock(512), MLPBlock(512), MLPBlock(512), MLPBlock(512),
    # )


    # prior_network = nn.Sequential(
    #     MemoryLayer('#0'),
    #     nn.Conv2d(in_channels * 2, 8, 1),
    #     # ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
    #     # MemoryLayer('#1'),
    #     # nn.AvgPool2d(2, 2),
    #     # ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
    #     # MemoryLayer('#2'),
    #     # nn.AvgPool2d(2, 2),
    #     ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
    #     MemoryLayer('#3'),
    #     nn.AvgPool2d(2, 2),
    #     ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
    #     MemoryLayer('#4'),
    #     nn.AvgPool2d(2, 2), nn.Conv2d(8, 16, 1),
    #     ResBlock(16, 8), ResBlock(16, 8), ResBlock(16, 8), ResBlock(16, 8),
    #     MemoryLayer('#5'),
    #     nn.AvgPool2d(2, 2), nn.Conv2d(16, 32, 1),
    #     ResBlock(32, 16), ResBlock(32, 16), ResBlock(32, 16), ResBlock(32, 16),
    #     MemoryLayer('#6'),
    #     nn.Conv2d(32, min(z_dim, 64), 1),
    #     # ResBlock(64, 32), ResBlock(64, 32), ResBlock(64, 32), ResBlock(64, 32),
    #     ResBlock(min(z_dim, 64), min(z_dim // 2, 32)), 
    #     ResBlock(min(z_dim, 64), min(z_dim // 2, 32)), 
    #     ResBlock(min(z_dim, 64), min(z_dim // 2, 32)), 
    #     ResBlock(min(z_dim, 64), min(z_dim // 2, 32)), 
    #     MemoryLayer('#7'),
    #     nn.Conv2d(min(z_dim, 64), min(z_dim, 128), 1),
    #     # nn.AvgPool2d(2, 2), nn.Conv2d(64, 128, 1),
    #     ResBlock(min(z_dim, 128), min(z_dim // 2, 64)), 
    #     ResBlock(min(z_dim, 128), min(z_dim // 2, 64)), 
    #     ResBlock(min(z_dim, 128), min(z_dim // 2, 64)), 
    #     ResBlock(min(z_dim, 128), min(z_dim // 2, 64)), 
    #     MemoryLayer('#8'),
    #     nn.Conv2d(min(z_dim, 128), min(z_dim, 256), 1),
    #     ResBlock(min(z_dim, 256), min(z_dim // 2, 128)), 
    #     ResBlock(min(z_dim, 256), min(z_dim // 2, 128)), 
    #     ResBlock(min(z_dim, 256), min(z_dim // 2, 128)), 
    #     ResBlock(min(z_dim, 256), min(z_dim // 2, 128)),
    #     MemoryLayer('#9'),
    #     nn.Conv2d(min(z_dim, 256), min(z_dim, 512), 1),
    #     # MLPBlock(512), MLPBlock(512), MLPBlock(512), MLPBlock(512),
    # )

    # # z_dim /= 2
    # generative_network = nn.Sequential(
    #     # nn.Conv2d(256, 256, 1),
    #     # MLPBlock(256), MLPBlock(256), MLPBlock(256), MLPBlock(256),
    #     nn.Conv2d(min(z_dim // 2, 256), min(z_dim // 2, 256), 1),
        
    #     # nn.Conv2d(256, 128, 1), nn.Upsample(scale_factor=3),
    #     # MemoryLayer('#9', True), nn.Conv2d(384, 128, 1),
    #     MemoryLayer('#9', True), 
    #     nn.Conv2d(min(z_dim, 256) + min(z_dim // 2, 256), min(z_dim // 2, 128), 1),

    #     ResBlock(min(z_dim // 2, 128), min(z_dim // 4, 64)), 
    #     ResBlock(min(z_dim // 2, 128), min(z_dim // 4, 64)), 
    #     ResBlock(min(z_dim // 2, 128), min(z_dim // 4, 64)), 
    #     ResBlock(min(z_dim // 2, 128), min(z_dim // 4, 64)), 
    #     nn.Conv2d(min(z_dim // 2, 128), min(z_dim // 2, 64), 1), #nn.Upsample(scale_factor=2),

    #     MemoryLayer('#8', True), 
    #     nn.Conv2d(min(z_dim, 128) + min(z_dim // 2, 64), min(z_dim // 2, 64), 1),
    #     ResBlock(min(z_dim // 2, 64), min(z_dim // 2, 32)), 
    #     ResBlock(min(z_dim // 2, 64), min(z_dim // 2, 32)), 
    #     ResBlock(min(z_dim // 2, 64), min(z_dim // 2, 32)), 
    #     ResBlock(min(z_dim // 2, 64), min(z_dim // 2, 32)),
    #     nn.Conv2d(min(z_dim // 2, 64), 32, 1), #nn.Upsample(scale_factor=2),

    #     MemoryLayer('#7', True), 
    #     nn.Conv2d(min(z_dim, 64) + 32, 32, 1),
    #     ResBlock(32, 16), ResBlock(32, 16), ResBlock(32, 16), ResBlock(32, 16),
    #     nn.Conv2d(32, 16, 1), #nn.Upsample(scale_factor=2),
        
    #     MemoryLayer('#6', True), nn.Conv2d(48, 16, 1),
    #     ResBlock(16, 8), ResBlock(16, 8), ResBlock(16, 8), ResBlock(16, 8),
    #     nn.Conv2d(16, 8, 1), nn.Upsample(scale_factor=2),
        
    #     MemoryLayer('#5', True), nn.Conv2d(24, 8, 1),
    #     ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
    #     nn.Upsample(scale_factor=2),
        
    #     MemoryLayer('#4', True), nn.Conv2d(16, 8, 1),
    #     ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
    #     nn.Upsample(scale_factor=2),
    #     MemoryLayer('#3', True), nn.Conv2d(16, 8, 1),
    #     ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
    #     # nn.Upsample(scale_factor=2),
    #     # MemoryLayer('#2', True), nn.Conv2d(16, 8, 1),
    #     # ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
    #     # nn.Upsample(scale_factor=2),
    #     # MemoryLayer('#1', True), nn.Conv2d(16, 8, 1),
    #     # ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
    #     MemoryLayer('#0', True), nn.Conv2d(8 + in_channels * 2, 8, 1),
    #     ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
    #     nn.Conv2d(8, in_channels * 2, 1),
    # )
    return proposal_network, prior_network, generative_network


def get_vae_networks(in_channels=1, out_channels=1, z_dim=64, metadata_channels=0, ch=16, discriminator=False):
    
    
    proposal_network = nn.Sequential(
        DownSampleResBlock(in_channels, ch, batchnorm=True),
        DownSampleResBlock(ch, ch * 2, batchnorm=True),
        DownSampleResBlock(ch * 2, ch * 4, batchnorm=True),
        DownSampleResBlock(ch * 4, z_dim, batchnorm=True)
    )

    generative_network = nn.Sequential(
        UpSampleResBlock(z_dim // 2, ch * 4),
        UpSampleResBlock(ch * 4, ch * 2),
        UpSampleResBlock(ch * 2, ch ),
        UpSampleResBlock(ch , out_channels)
    )



    discriminator_network = None
    if discriminator:
        discriminator_network = nn.Sequential(
            DownSampleResBlock(in_channels, ch, batchnorm=False),
            DownSampleResBlock(ch, ch * 2, batchnorm=False),
            DownSampleResBlock(ch * 2, ch * 4, batchnorm=False),
            DownSampleResBlock(ch * 4, ch * 4, batchnorm=False),
            nn.Conv2d(ch * 4, 1, kernel_size=16)
        )
    return proposal_network, generative_network, discriminator_network

# def get_vae_networks2(in_channels=1, out_channels=1, z_dim=64, metadata_channels=0):

#     proposal_network = nn.Sequential(
#         nn.Conv2d(in_channels, 8, 1),
#         # ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
#         # nn.AvgPool2d(2, 2),
#         # ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
#         # nn.AvgPool2d(2, 2),
#         ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
#         nn.AvgPool2d(2, 2),
#         ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
#         nn.AvgPool2d(2, 2), nn.Conv2d(8, 16, 1),
#         ResBlock(16, 8), ResBlock(16, 8), ResBlock(16, 8), ResBlock(16, 8),
#         nn.AvgPool2d(2, 2), nn.Conv2d(16, 32, 1),
#         ResBlock(32, 16), ResBlock(32, 16), ResBlock(32, 16), ResBlock(32, 16),
#         nn.AvgPool2d(2, 2), nn.Conv2d(32, 64, 1),
#         ResBlock(64, 32), ResBlock(64, 32), ResBlock(64, 32), ResBlock(64, 32),
#         nn.Conv2d(64, 128, 1),
#         ResBlock(128, 64), ResBlock(128, 64),
#         ResBlock(128, 64), ResBlock(128, 64),
#         nn.Conv2d(128, 256, 1),
#         ResBlock(256, 128), ResBlock(256, 128),
#         ResBlock(256, 128), ResBlock(256, 128),
#         nn.Conv2d(256, z_dim, 1),
#         # MLPBlock(512), MLPBlock(512), MLPBlock(512), MLPBlock(512),
#     )


#     generative_network = nn.Sequential(
#         nn.Conv2d(z_dim // 2 + metadata_channels, 256, 1),
#         MLPBlock(256), MLPBlock(256), MLPBlock(256), MLPBlock(256),
#         nn.Conv2d(256, 128, 1),# nn.Upsample(scale_factor=3),
#         # MemoryLayer('#9', True), nn.Conv2d(384, 128, 1),
#         ResBlock(128, 64), ResBlock(128, 64),
#         ResBlock(128, 64), ResBlock(128, 64),
#         nn.Conv2d(128, 64, 1),# nn.Upsample(scale_factor=2),
#         # MemoryLayer('#8', True), nn.Conv2d(192, 64, 1),
#         ResBlock(64, 32), ResBlock(64, 32), ResBlock(64, 32), ResBlock(64, 32),
#         nn.Conv2d(64, 32, 1),# nn.Upsample(scale_factor=2),
#         # MemoryLayer('#7', True), nn.Conv2d(96, 32, 1),
#         ResBlock(32, 16), ResBlock(32, 16), ResBlock(32, 16), ResBlock(32, 16),
#         nn.Conv2d(32, 16, 1), nn.Upsample(scale_factor=2),
#         # MemoryLayer('#6', True), nn.Conv2d(48, 16, 1),
#         ResBlock(16, 8), ResBlock(16, 8), ResBlock(16, 8), ResBlock(16, 8),
#         nn.Conv2d(16, 8, 1), nn.Upsample(scale_factor=2),
#         # MemoryLayer('#5', True), nn.Conv2d(24, 8, 1),
#         ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
#         nn.Upsample(scale_factor=2),
#         # MemoryLayer('#4', True), nn.Conv2d(16, 8, 1),
#         ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
#         nn.Upsample(scale_factor=2),
#         # MemoryLayer('#3', True), nn.Conv2d(16, 8, 1),
#         ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
#         # nn.Upsample(scale_factor=2),
#         # MemoryLayer('#2', True), nn.Conv2d(16, 8, 1),
#         # ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
#         # nn.Upsample(scale_factor=2),
#         # MemoryLayer('#1', True), nn.Conv2d(16, 8, 1),
#         # ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
#         # MemoryLayer('#0', True), nn.Conv2d(10, 8, 1),
#         # ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
#         nn.Conv2d(8, out_channels, 1),
#     )


#     return proposal_network, generative_network


    

def get_vae_networks3(in_channels=1, out_channels=1, z_dim=64, metadata_channels=0, discriminator=False):


    in_c = in_channels
    if metadata_channels:
        in_c *= 2

    print("expected input channels", in_c)
    proposal_network = nn.Sequential(
        nn.Conv2d(in_c, 16, 1),
        ResBlock(16, 8), ResBlock(16, 8),
        nn.AvgPool2d(2, 2),
        ResBlock(16, 8), ResBlock(16, 8),
        nn.AvgPool2d(2, 2),
        ResBlock(16, 8), ResBlock(16, 8),
        nn.AvgPool2d(2, 2), nn.Conv2d(16, 32, 1),
        ResBlock(32, 16), ResBlock(32, 16),
        nn.AvgPool2d(2, 2), nn.Conv2d(32, 64, 1),
        ResBlock(64, 32), ResBlock(64, 32),
        nn.Conv2d(64, 128, 1),
        ResBlock(128, 64), ResBlock(128, 64),
        nn.Conv2d(128, 256, 1),
        ResBlock(256, 128), ResBlock(256, 128),
        nn.Conv2d(256, z_dim, 1),
    )

    generative_network = nn.Sequential(
        # 16 x 16 x 32
        nn.Conv2d(z_dim // 2 + metadata_channels, 256, 1), 
        # MLPBlock(256), MLPBlock(256),
        nn.Conv2d(256, 128, 1),
        ResBlock(128, 64), ResBlock(128, 64),
        nn.Conv2d(128, 64, 1),
        ResBlock(64, 32), ResBlock(64, 32),
        nn.Conv2d(64, 32, 1),
        ResBlock(32, 16), ResBlock(32, 16),
        nn.Conv2d(32, 16, 1), nn.Upsample(scale_factor=2),
        ResBlock(16, 8), ResBlock(16, 8),
        nn.Upsample(scale_factor=2),
        ResBlock(16, 8), ResBlock(16, 8),
        nn.Upsample(scale_factor=2),
        ResBlock(16, 8), ResBlock(16, 8),
        nn.Upsample(scale_factor=2),
        ResBlock(16, 8), ResBlock(16, 8),
        nn.Conv2d(16, out_channels, 1),
    )

    discriminator_network = None
    if discriminator:
        discriminator_network = nn.Sequential(
            nn.Conv2d(in_channels, 16, 1),
            ResBlock(16, 8), ResBlock(16, 8),
            nn.AvgPool2d(2, 2),
            ResBlock(16, 8), ResBlock(16, 8),
            nn.AvgPool2d(2, 2),
            ResBlock(16, 8), ResBlock(16, 8),
            nn.AvgPool2d(2, 2), nn.Conv2d(16, 32, 1),
            ResBlock(32, 16), ResBlock(32, 16),
            nn.AvgPool2d(2, 2), nn.Conv2d(32, 64, 1),
            ResBlock(64, 32), ResBlock(64, 32),
            nn.Conv2d(64, 128, 1),
            ResBlock(128, 64), ResBlock(128, 64),
            nn.Conv2d(128, 256, 1),
            ResBlock(256, 128), ResBlock(256, 128),
            nn.Conv2d(256, 1, kernel_size=16)
        )
    return proposal_network, generative_network, discriminator_network

    
def get_vae_networks4(in_channels=1, out_channels=1, z_dim=64, metadata_channels=0, discriminator=False):

    in_c = in_channels
    if metadata_channels:
        in_c *= 2
    proposal_network = nn.Sequential(
        nn.Conv2d(in_c, 16, 1),
        ResBlock(16, 8), ResBlock(16, 8),
        ResBlock(16, 8), ResBlock(16, 8),
        nn.AvgPool2d(2, 2),

        ResBlock(16, 8), ResBlock(16, 8),
        nn.Conv2d(16, 32, 1),
        ResBlock(32, 16), ResBlock(32, 16),
        nn.AvgPool2d(2, 2), nn.Conv2d(32, 64, 1),

        ResBlock(64, 32), ResBlock(64, 32),
        nn.AvgPool2d(2, 2), nn.Conv2d(64, 128, 1),
        
        ResBlock(128, 64), ResBlock(128, 64),
        nn.AvgPool2d(2, 2), nn.Conv2d(128, 256, 1),

        ResBlock(256, 128), ResBlock(256, 128),
        nn.Conv2d(256, z_dim, 1),
    )

    generative_network = nn.Sequential(
        # 16 x 16 x 32
        nn.Conv2d(z_dim // 2 + metadata_channels, 256, 1), 
        # MLPBlock(256), MLPBlock(256),
        nn.Conv2d(256, 128, 1),
        ResBlock(128, 64), ResBlock(128, 64),
        nn.Conv2d(128, 64, 1), nn.Upsample(scale_factor=2),

        ResBlock(64, 32), ResBlock(64, 32),
        nn.Conv2d(64, 32, 1), nn.Upsample(scale_factor=2),

        ResBlock(32, 16), ResBlock(32, 16),
        nn.Conv2d(32, 16, 1), nn.Upsample(scale_factor=2),
        
        ResBlock(16, 8), ResBlock(16, 8),
        ResBlock(16, 8), ResBlock(16, 8),
        nn.Upsample(scale_factor=2),
        ResBlock(16, 8), ResBlock(16, 8),
        ResBlock(16, 8), ResBlock(16, 8),
        nn.Conv2d(16, out_channels, 1),
    )

    discriminator_network = None
    if discriminator:
        discriminator_network = nn.Sequential(
            nn.Conv2d(in_channels, 16, 1),
            ResBlock(16, 8), ResBlock(16, 8),
            nn.AvgPool2d(2, 2),
            ResBlock(16, 8), ResBlock(16, 8),
            nn.AvgPool2d(2, 2),
            ResBlock(16, 8), ResBlock(16, 8),
            nn.AvgPool2d(2, 2), nn.Conv2d(16, 32, 1),
            ResBlock(32, 16), ResBlock(32, 16),
            nn.AvgPool2d(2, 2), nn.Conv2d(32, 64, 1),
            ResBlock(64, 32), ResBlock(64, 32),
            nn.Conv2d(64, 128, 1),
            ResBlock(128, 64), ResBlock(128, 64),
            nn.Conv2d(128, 256, 1),
            ResBlock(256, 128), ResBlock(256, 128),
            nn.Conv2d(256, 1, kernel_size=16)
        )
    return proposal_network, generative_network, discriminator_network

    # proposal_network = nn.Sequential(
    #     nn.Conv2d(in_channels, 16, 1),
    #     ResBlock(16, 8), ResBlock(16, 8),
    #     nn.AvgPool2d(2, 2),
    #     ResBlock(16, 8), ResBlock(16, 8),
    #     nn.AvgPool2d(2, 2),
    #     ResBlock(16, 8), ResBlock(16, 8),
    #     nn.AvgPool2d(2, 2), nn.Conv2d(16, 32, 1),
    #     ResBlock(32, 16), ResBlock(32, 16),
    #     nn.Conv2d(32, 64, 1),
    #     ResBlock(64, 32), ResBlock(64, 32),
    #     nn.Conv2d(64, 128, 1),
    #     ResBlock(128, 64), ResBlock(128, 64),
    #     nn.Conv2d(128, 256, 1),
    #     ResBlock(256, 128), ResBlock(256, 128),
    #     nn.Conv2d(256, z_dim, 1),
    # )

    # generative_network = nn.Sequential(
    #     nn.Conv2d(z_dim // 2 + metadata_channels, 256, 1),
    #     MLPBlock(256), MLPBlock(256),
    #     nn.Conv2d(256, 128, 1),
    #     ResBlock(128, 64), ResBlock(128, 64),
    #     nn.Conv2d(128, 64, 1),
    #     ResBlock(64, 32), ResBlock(64, 32),
    #     nn.Conv2d(64, 32, 1),
    #     ResBlock(32, 16), ResBlock(32, 16),
    #     nn.Conv2d(32, 16, 1),
    #     ResBlock(16, 8), ResBlock(16, 8),
    #     nn.Upsample(scale_factor=2),
    #     ResBlock(16, 8), ResBlock(16, 8),
    #     nn.Upsample(scale_factor=2),
    #     ResBlock(16, 8), ResBlock(16, 8),
    #     nn.Upsample(scale_factor=2),
    #     ResBlock(16, 8), ResBlock(16, 8),
    #     nn.Conv2d(16, out_channels, 1),
    # )

    # return proposal_network, generative_network

def get_vae_networks5(in_channels=1, out_channels=1, z_dim=64, metadata_channels=0, discriminator=False):

    in_c = in_channels
    if metadata_channels:
        in_c *= 2
    proposal_network = nn.Sequential(
        nn.Conv2d(in_c, 16, 1),
        ResBlock(16, 16), ResBlock(16, 16),
        nn.AvgPool2d(2, 2),
        ResBlock(16, 16), ResBlock(16, 16),
        nn.AvgPool2d(2, 2),
        ResBlock(16, 16), ResBlock(16, 16),
        nn.AvgPool2d(2, 2), nn.Conv2d(16, 32, 1),
        ResBlock(32, 32), ResBlock(32, 32),
        nn.AvgPool2d(2, 2), nn.Conv2d(32, 64, 1),
        ResBlock(64, 64), ResBlock(64, 64),
        nn.Conv2d(64, 128, 1),
        ResBlock(128, 128), ResBlock(128, 128),
        nn.Conv2d(128, 256, 1),
        ResBlock(256, 256), ResBlock(256, 256),
        nn.Conv2d(256, z_dim, 1),
    )

    generative_network = nn.Sequential(
        # 16 x 16 x 32
        nn.Conv2d(z_dim // 2 + metadata_channels, 256, 1), 
        # MLPBlock(256), MLPBlock(256),
        nn.Conv2d(256, 128, 1),
        ResBlock(128, 128), ResBlock(128, 128),
        nn.Conv2d(128, 64, 1),
        ResBlock(64, 64), ResBlock(64, 64),
        nn.Conv2d(64, 32, 1),
        ResBlock(32, 32), ResBlock(32, 32),
        nn.Conv2d(32, 16, 1), nn.Upsample(scale_factor=2),
        ResBlock(16, 16), ResBlock(16, 16),
        nn.Upsample(scale_factor=2),
        ResBlock(16, 16), ResBlock(16, 16),
        nn.Upsample(scale_factor=2),
        ResBlock(16, 16), ResBlock(16, 16),
        nn.Upsample(scale_factor=2),
        ResBlock(16, 16), ResBlock(16, 16),
        nn.Conv2d(16, out_channels, 1),
    )

    discriminator_network = None
    if discriminator:
        discriminator_network = nn.Sequential(
            nn.Conv2d(in_channels, 16, 1),
            ResBlock(16, 8), ResBlock(16, 8),
            nn.AvgPool2d(2, 2),
            ResBlock(16, 8), ResBlock(16, 8),
            nn.AvgPool2d(2, 2),
            ResBlock(16, 8), ResBlock(16, 8),
            nn.AvgPool2d(2, 2), nn.Conv2d(16, 32, 1),
            ResBlock(32, 16), ResBlock(32, 16),
            nn.AvgPool2d(2, 2), nn.Conv2d(32, 64, 1),
            ResBlock(64, 32), ResBlock(64, 32),
            nn.Conv2d(64, 128, 1),
            ResBlock(128, 64), ResBlock(128, 64),
            nn.Conv2d(128, 256, 1),
            ResBlock(256, 128), ResBlock(256, 128),
            nn.Conv2d(256, 1, kernel_size=16)
        )
    return proposal_network, generative_network, discriminator_network

    
def get_vae_networks6(in_channels=1, out_channels=1, z_dim=64, metadata_channels=0, discriminator=False):


    # upsampling and residual block inner dim fixed.
    in_c = in_channels
    if metadata_channels:
        in_c *= 2
    proposal_network = nn.Sequential(
        nn.Conv2d(in_c, 16, 1),
        ResBlock(16, 16), ResBlock(16, 16),
        ResBlock(16, 16), ResBlock(16, 16),
        nn.AvgPool2d(2, 2),

        ResBlock(16, 16), ResBlock(16, 16),
        nn.Conv2d(16, 32, 1),
        ResBlock(32, 32), ResBlock(32, 32),
        nn.AvgPool2d(2, 2), nn.Conv2d(32, 64, 1),

        ResBlock(64, 64), ResBlock(64, 64),
        nn.AvgPool2d(2, 2), nn.Conv2d(64, 128, 1),
        
        ResBlock(128, 128), ResBlock(128, 128),
        nn.AvgPool2d(2, 2), nn.Conv2d(128, 256, 1),

        ResBlock(256, 256), ResBlock(256, 256),
        nn.Conv2d(256, z_dim, 1),
    )

    generative_network = nn.Sequential(
        # 16 x 16 x 32
        nn.Conv2d(z_dim // 2 + metadata_channels, 256, 1), 
        # MLPBlock(256), MLPBlock(256),
        nn.Conv2d(256, 128, 1),
        ResBlock(128, 128), ResBlock(128, 128),
        nn.Conv2d(128, 64, 1), nn.Upsample(scale_factor=2),

        ResBlock(64, 64), ResBlock(64, 64),
        nn.Conv2d(64, 32, 1), nn.Upsample(scale_factor=2),

        ResBlock(32, 32), ResBlock(32, 32),
        nn.Conv2d(32, 16, 1), nn.Upsample(scale_factor=2),
        
        ResBlock(16, 16), ResBlock(16, 16),
        ResBlock(16, 16), ResBlock(16, 16),
        nn.Upsample(scale_factor=2),
        ResBlock(16, 16), ResBlock(16, 16),
        ResBlock(16, 16), ResBlock(16, 16),
        nn.Conv2d(16, out_channels, 1),
    )

    discriminator_network = None
    if discriminator:
        discriminator_network = nn.Sequential(
            nn.Conv2d(in_channels, 16, 1),
            ResBlock(16, 8), ResBlock(16, 8),
            nn.AvgPool2d(2, 2),
            ResBlock(16, 8), ResBlock(16, 8),
            nn.AvgPool2d(2, 2),
            ResBlock(16, 8), ResBlock(16, 8),
            nn.AvgPool2d(2, 2), nn.Conv2d(16, 32, 1),
            ResBlock(32, 16), ResBlock(32, 16),
            nn.AvgPool2d(2, 2), nn.Conv2d(32, 64, 1),
            ResBlock(64, 32), ResBlock(64, 32),
            nn.Conv2d(64, 128, 1),
            ResBlock(128, 64), ResBlock(128, 64),
            nn.Conv2d(128, 256, 1),
            ResBlock(256, 128), ResBlock(256, 128),
            nn.Conv2d(256, 1, kernel_size=16)
        )
    return proposal_network, generative_network, discriminator_network
    

def get_vae_networks7(in_channels=1, out_channels=1, z_dim=64, metadata_channels=0, discriminator=False):


    # upsampling and residual block inner dim fixed.
    in_c = in_channels
    proposal_network = nn.ModuleList([
        nn.Conv2d(in_c, 16, 1),
        CResBlock(16, 16, n_cond=metadata_channels), CResBlock(16, 16, n_cond=metadata_channels),
        CResBlock(16, 16, n_cond=metadata_channels), CResBlock(16, 16, n_cond=metadata_channels),
        nn.AvgPool2d(2, 2),

        CResBlock(16, 16, n_cond=metadata_channels), CResBlock(16, 16, n_cond=metadata_channels),
        nn.Conv2d(16, 32, 1),
        CResBlock(32, 32, n_cond=metadata_channels), CResBlock(32, 32, n_cond=metadata_channels),
        nn.AvgPool2d(2, 2), nn.Conv2d(32, 64, 1),

        CResBlock(64, 64, n_cond=metadata_channels), CResBlock(64, 64, n_cond=metadata_channels),
        nn.AvgPool2d(2, 2), nn.Conv2d(64, 128, 1),
        
        CResBlock(128, 128, n_cond=metadata_channels), CResBlock(128, 128, n_cond=metadata_channels),
        nn.AvgPool2d(2, 2), nn.Conv2d(128, 256, 1),

        CResBlock(256, 256, n_cond=metadata_channels), CResBlock(256, 256, n_cond=metadata_channels),
        nn.Conv2d(256, z_dim, 1),
    ])

    generative_network = nn.ModuleList([
        # 16 x 16 x 32
        nn.Conv2d(z_dim // 2, 256, 1), 
        # MLPBlock(256), MLPBlock(256),
        nn.Conv2d(256, 128, 1),
        CResBlock(128, 128, n_cond=metadata_channels), CResBlock(128, 128, n_cond=metadata_channels),
        nn.Conv2d(128, 64, 1), nn.Upsample(scale_factor=2),

        CResBlock(64, 64, n_cond=metadata_channels), CResBlock(64, 64, n_cond=metadata_channels),
        nn.Conv2d(64, 32, 1), nn.Upsample(scale_factor=2),

        CResBlock(32, 32, n_cond=metadata_channels), CResBlock(32, 32, n_cond=metadata_channels),
        nn.Conv2d(32, 16, 1), nn.Upsample(scale_factor=2),
        
        CResBlock(16, 16, n_cond=metadata_channels), CResBlock(16, 16, n_cond=metadata_channels),
        CResBlock(16, 16, n_cond=metadata_channels), CResBlock(16, 16, n_cond=metadata_channels),
        nn.Upsample(scale_factor=2),
        CResBlock(16, 16, n_cond=metadata_channels), CResBlock(16, 16, n_cond=metadata_channels),
        CResBlock(16, 16, n_cond=metadata_channels), CResBlock(16, 16, n_cond=metadata_channels),
        nn.Conv2d(16, out_channels, 1),
    ])

    discriminator_network = None
    
    return proposal_network, generative_network, discriminator_network
    # exact architecture of anovaegan
    # proposal_network = nn.Sequential(
    #     nn.Conv2d(in_channels, 16, stride=2, padding=2, kernel_size=5), # 128x128x16
    #     nn.LeakyReLU(),
    #     nn.Conv2d(16, 32, stride=2, padding=2, kernel_size=5),# 64x64x32
    #     nn.LeakyReLU(),
    #     nn.Conv2d(32, 64, stride=2, padding=2, kernel_size=5),# 32x32x64
    #     nn.LeakyReLU(),
    #     nn.Conv2d(64, 128, stride=2, padding=2, kernel_size=5),# 16x16x128
    #     nn.LeakyReLU(),
    #     nn.Conv2d(128, 16, stride=1, padding=0, kernel_size=1), # 16x16x z_dim
    #     nn.LeakyReLU(),
    #     nn.Conv2d(16, z_dim, stride=1, padding=0, kernel_size=1)

    # )

    # generative_network = nn.Sequential(
    #     nn.Conv2d(z_dim // 2, 16, stride=1, padding=1, kernel_size=3), # 16x16x16
    #     nn.LeakyReLU(),
    #     nn.Conv2d(16, 128, stride=1, padding=0, kernel_size=1), # 16x16x128
    #     nn.LeakyReLU(),
    #     nn.ConvTranspose2d(128, 64, stride=2, padding=1, kernel_size=4),# 32x32x64
    #     nn.LeakyReLU(),
    #     nn.ConvTranspose2d(64, 32, stride=2, padding=1, kernel_size=4),# 64x64x32
    #     nn.LeakyReLU(),
    #     nn.ConvTranspose2d(32, 16, stride=2, padding=1, kernel_size=4),# 128x128x16
    #     nn.LeakyReLU(),
    #     nn.ConvTranspose2d(16, 16, stride=2, padding=1, kernel_size=4), # 256x256x16
    #     nn.LeakyReLU(),
    #     nn.Conv2d(16, out_channels, stride=1, padding=0, kernel_size=1) # 16x16x128
    # )

    # return proposal_network, generative_network

def get_dense_vae_networks(in_channels=1, z_dim=256):

    proposal_network = nn.Sequential(
        nn.Conv2d(in_channels, 8, 1),
        # ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
        # nn.AvgPool2d(2, 2),
        # ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
        # nn.AvgPool2d(2, 2),
        ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
        nn.AvgPool2d(2, 2),
        ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
        nn.AvgPool2d(2, 2), nn.Conv2d(8, 16, 1),
        ResBlock(16, 8), ResBlock(16, 8), ResBlock(16, 8), ResBlock(16, 8),
        nn.AvgPool2d(2, 2), nn.Conv2d(16, 32, 1),
        ResBlock(32, 16), ResBlock(32, 16), ResBlock(32, 16), ResBlock(32, 16),
        nn.AvgPool2d(2, 2), nn.Conv2d(32, 64, 1),
        ResBlock(64, 32), ResBlock(64, 32), ResBlock(64, 32), ResBlock(64, 32),
        nn.AvgPool2d(2, 2), nn.Conv2d(64, 128, 1),
        ResBlock(128, 64), ResBlock(128, 64),
        ResBlock(128, 64), ResBlock(128, 64),
        nn.AvgPool2d(2, 2), nn.Conv2d(128, 256, 1),
        ResBlock(256, 128), ResBlock(256, 128),
        ResBlock(256, 128), ResBlock(256, 128),
        nn.AvgPool2d(2, 2), nn.Conv2d(256, 512, 1),
        MLPBlock(512), MLPBlock(512), MLPBlock(512), MLPBlock(512),
        nn.Conv2d(512, z_dim, 1),
    )


    generative_network = nn.Sequential(
        nn.Conv2d(z_dim // 2, 256, 1),
        MLPBlock(256), MLPBlock(256), MLPBlock(256), MLPBlock(256),
        nn.Conv2d(256, 128, 1), nn.Upsample(scale_factor=3),
        # MemoryLayer('#9', True), nn.Conv2d(384, 128, 1),
        ResBlock(128, 64), ResBlock(128, 64),
        ResBlock(128, 64), ResBlock(128, 64),
        nn.Conv2d(128, 64, 1), nn.Upsample(scale_factor=2),
        # MemoryLayer('#8', True), nn.Conv2d(192, 64, 1),
        ResBlock(64, 32), ResBlock(64, 32), ResBlock(64, 32), ResBlock(64, 32),
        nn.Conv2d(64, 32, 1), nn.Upsample(scale_factor=2),
        # MemoryLayer('#7', True), nn.Conv2d(96, 32, 1),
        ResBlock(32, 16), ResBlock(32, 16), ResBlock(32, 16), ResBlock(32, 16),
        nn.Conv2d(32, 16, 1), nn.Upsample(scale_factor=2),
        # MemoryLayer('#6', True), nn.Conv2d(48, 16, 1),
        ResBlock(16, 8), ResBlock(16, 8), ResBlock(16, 8), ResBlock(16, 8),
        nn.Conv2d(16, 8, 1), nn.Upsample(scale_factor=2),
        # MemoryLayer('#5', True), nn.Conv2d(24, 8, 1),
        ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
        nn.Upsample(scale_factor=2),
        # MemoryLayer('#4', True), nn.Conv2d(16, 8, 1),
        ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
        nn.Upsample(scale_factor=2),
        # MemoryLayer('#3', True), nn.Conv2d(16, 8, 1),
        ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
        # nn.Upsample(scale_factor=2),
        # MemoryLayer('#2', True), nn.Conv2d(16, 8, 1),
        # ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
        # nn.Upsample(scale_factor=2),
        # MemoryLayer('#1', True), nn.Conv2d(16, 8, 1),
        # ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
        # MemoryLayer('#0', True), nn.Conv2d(10, 8, 1),
        # ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
        nn.Conv2d(8, 2 * in_channels, 1),
    )
    return proposal_network, generative_network


