# encoder portion (Res-Net Encoder Model)

# https://wandb.ai/amanarora/Written-Reports/reports/Understanding-ResNets-A-Deep-Dive-into-Residual-Networks-with-PyTorch--Vmlldzo1MDAxMTk5
# https://www.youtube.com/watch?v=HBYQvKlaE0A

# GM12878 1/16 downsampling


import torch
from torch import nn # neural network

# inherit neural network module
class ResidualBlock(nn.Module):

  # in channels is num of channels into residual block
  # out channels is num of channels out of residual block
  # stride is movement for convultional filter -> 1 unit at a time
  def __init__(self, in_channels, out_channels, stride=1):
    super().__init__()

    # first layer so dimensions not fixed
    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False) # convert input from in channels to out channels
    # self.bn1 = nn.BatchNorm2d(out_channels) no batch normalization
    self.relu = nn.ReLU(inplace=True) # relu activation layer

    # second layer, dimension is fixed
    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)   # second layer outchannels as input and then to output is also out_channels

  def forward(self, x):
    # following pattern from image from https://arxiv.org/abs/1707.02921
    out = self.conv1(x)
    out = self.relu(out)
    out = self.conv2(out)
    x1 = x + out * 0.1 # the paper recommends a scaling factor of 0.1, possibly change
    return out

class HiCEncoder(nn.Module):

  # hic has only interaction frequencies, so singular channel. in channels = 1
  # base_channels (64 for most super res VAEs)
  # res_blocks = 16 for base ESDR models
  # latent dim = 128 for starting
  # image size 40, based on hicplus sizes.
  def __init__(self, in_channels=1, base_channels=64, num_res_blocks=16, latent_dim = 128, image_size = 40):

    super(HiCEncoder, self).__init__()

    # starting convultion layer
    self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False)

    # res block stack
    res_blocks = []
    for _ in range(num_res_blocks):
      res_blocks.append(ResidualBlock(base_channels, base_channels))

    self.res_blocks = nn.Sequential(*res_blocks) #

    # layer after res blocks
    self.conv2 = nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False)

    # num features after layers
    num_features = base_channels * image_size * image_size

    # linear layer to latent space "The later layers of the encoder flatten the output from the residual network and output the standard deviation (σ) and mean (μ) which are the outputs of a dense layer"
    self.fc_mu = nn.Linear(self.feature_size, latent_dim) # mean of latent space distruburion
    self.fc_sigma = nn.Linear(self.feature_size, latent_dim) # std of latent space distribution


  def reparam_trick(self, mu, sigma):
    epsilon = torch.randn_like(mu) # sample from gaussian distribution w same shape as mu

    # reparam formula from paper
    z = mu + sigma * epsilon
    return z


  def forward(self, x):

    # init convultion
    x = self.conv1(x)

    # pass thru the res blocks
    x = self.res_blocks(x)

    # second convultion
    x = self.conv2(x)

    # flatten
    x = x.view(x.size(0), -1)

    # latent mean + std
    mu = self.fc_mu(x)
    raw_sigma = self.fc_sigma(x)
    # sigma must be positive?
    sigma = torch.exp(raw_sigma)

    # reparam for sampling
    z = self.reparameterize(mu, sigma)
    return z, mu, sigma


class HiCDecoder(nn.Module):

    def __init__(self, base_channels=64, out_channels = 1, num_res_blocks=16, latent_dim = 128, image_size = 40, scale = 16):
      super(HiCDecoder, self).__init__()

      # latent space into original feature map
      self.fc = nn.Linear(latent_dim, base_channels * image_size * image_size)

      # same res-net as encoder
      res_blocks = []
      for _ in range(num_res_blocks):
            res_blocks.append(ResidualBlock(base_channels, base_channels))

      self.res_blocks = nn.Sequential(*res_blocks)

      # same res-net before upsampling
      self.conv_pre_up = nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False)

      # upsampling & pixel shuffling (dont get this stuff)
      self.conv_up = nn.Conv2d(base_channels, out_channels * (scale ** 2), kernel_size=3, stride=1, padding=1, bias=False)
      self.pixel_shuffle = nn.PixelShuffle(scale)


      # final conv for last layer
      self.conv_out = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, z):
      batch_size = z.size(0)

      # latent space to feature map
      x = self.fc(z)
      x = x.view(batch_size, self.base_channels, self.image_size, self.image_size)

      # residual blocks + first conv layer
      x = self.res_blocks(x)
      x = self.conv_pre_up(x)

      # convultion up & pixel shuffle
      x = self.conv_up(x)
      x = self.pixel_shuffle(x)

      # final layer
      x = self.conv_out(x)

      return x


class HiCVAE(nn.Module):

    def __init__(self, in_channels=1, base_channels=64, num_res_blocks=16, latent_dim=128, image_size=40, scale=16, out_channels=1):

        super(HiCVAE, self).__init__()
        self.encoder = HiCEncoder(in_channels, base_channels, num_res_blocks, latent_dim, image_size)
        self.decoder = HiCDecoder(base_channels, out_channels, num_res_blocks, latent_dim, image_size, scale)


    def forward(self, x):
        z, mu, sigma = self.encoder(x)
        recon = self.decoder(z)
        return recon, mu, sigma
