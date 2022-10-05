import torch.nn.functional as F  # not sure if we can just use torch.nn.ReLu()
import torch
import torch.nn as nn
import numpy as np

#-----------------------------Custom Layers------------------------------------#

class Norm_Layer(nn.Module):
    """
    Normalize the normal image to a vector map - 
    custom autograd function because we use a small epsilon to avoid dividing zero,
    (actually the pytorch autograd should also work?)
    """

    def __init__(self):
        super(Norm_Layer, self).__init__()

    def forward(self, input_normal):
        return NormalizeNormalFunction.apply(input_normal)


class NormalizeNormalFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_normal):
        in_shape = input_normal.shape
        normal = 2.0 * input_normal - 1.0

        # normalize the normal vector
        norm = torch.norm(normal, dim=1)
        norm = norm.view(in_shape[0], 1, in_shape[2], in_shape[3])
        norm = norm.repeat(1, in_shape[1], 1, 1) + 1e-8
        normal_normalized = torch.div(normal, norm)

        # ctx is a context object that can be used to stash information
        # for backward computation
        ctx.save_for_backward(input_normal, normal_normalized)

        return normal_normalized

    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input_normal, normal_normalized = ctx.saved_tensors
        grad_input = None

        sz = input_normal.shape
        normal = 2.0 * input_normal - 1.0
        sc = torch.sqrt(torch.sum(normal*normal, dim=1, keepdim=True)) + 1e-8
        sc = sc.repeat(1, sz[1], 1, 1)  # np.tile(sc,(1,sz[1],1,1))

        grad_input = torch.zeros(sz, dtype=torch.float32).to('cuda')
        for i in range(0, sz[0]):
            Ey = grad_output[i, ...]
            Ny = normal_normalized[i, ...]
            # np.sum(np.multiply(Ey,top[0].data[i,...]),axis=0,keepdims=True)
            ip = torch.sum(Ey * Ny, dim=0, keepdim=True)
            ip = ip.repeat(sz[1], 1, 1)  # np.tile(ip,(sz[1],1,1))
            # 2*np.divide(Ey - normal_normalized[i,...]*ip,sc[i,...])
            grad_input[i, ...] = 2*torch.div(Ey - Ny*ip, sc[i, ...])
        return grad_input


class Shading_Layer(nn.Module):
    """Calculate shading = normal * sh2 lighting"""

    def __init__(self):
        super(Shading_Layer, self).__init__()

    def forward(self, light, normal):
        '''
        get SH basis based on normal
        normal is a Nx3 matrix
        return a Nx9 matrix
        The order of SH here is:
        1, Y, Z, X, YX, YZ, 3Z^2-1, XZ, X^2-y^2
        '''
        in_shape = normal.shape
        shading = torch.zeros(in_shape, dtype=torch.float32).to('cuda')
        for i in range(0, normal.shape[0]):
            norm_3pix = normal[i, ...].view(in_shape[1], -1)

            norm_X = norm_3pix[0, :]
            norm_Y = norm_3pix[1, :]
            norm_Z = norm_3pix[2, :]

            sh_basis = torch.zeros(
                (in_shape[2]*in_shape[3], 9), dtype=torch.float32).to('cuda')
            att = np.pi*np.array([1, 2.0/3.0, 1/4.0])
            sh_basis[:, 0] = 0.5/np.sqrt(np.pi)*att[0]

            sh_basis[:, 1] = np.sqrt(3)/2/np.sqrt(np.pi)*norm_Y*att[1]
            sh_basis[:, 2] = np.sqrt(3)/2/np.sqrt(np.pi)*norm_Z*att[1]
            sh_basis[:, 3] = np.sqrt(3)/2/np.sqrt(np.pi)*norm_X*att[1]

            sh_basis[:, 4] = np.sqrt(15)/2/np.sqrt(np.pi)*norm_Y*norm_X*att[2]
            sh_basis[:, 5] = np.sqrt(15)/2/np.sqrt(np.pi)*norm_Y*norm_Z*att[2]
            sh_basis[:, 6] = np.sqrt(5)/4/np.sqrt(np.pi)*(3*norm_Z**2-1)*att[2]
            sh_basis[:, 7] = np.sqrt(15)/2/np.sqrt(np.pi)*norm_X*norm_Z*att[2]
            sh_basis[:, 8] = np.sqrt(
                15)/4/np.sqrt(np.pi)*(norm_X**2-norm_Y**2)*att[2]
            shading[i] = torch.matmul(sh_basis, light[i].view(in_shape[1], 9).T).T.view(
                in_shape[1], in_shape[2], in_shape[3])
        return shading


#-----------------------------Model Components---------------------------------#


class ConvBNReluBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBNReluBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, affine=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class OutputConvBlock(nn.Module):
    def __init__(self):
        super(OutputConvBlock, self).__init__()
        # conv: c128(k1)
        self.cbr6 = ConvBNReluBlock(128, 128, 1, stride=1, padding=0)
        # conv: c64(k3)
        self.cbr7 = ConvBNReluBlock(128, 64, 3, stride=1, padding=1)
        # output: c*3(k1)
        self.out = nn.Conv2d(64, 3, 1, stride=1, padding=0)

    def forward(self, x):
        # deconv bilinear upsampling
        x = nn.functional.interpolate(
            x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.cbr6(x)
        x = self.cbr7(x)
        x = self.out(x)
        return x


class OutputConvBlock_specularmap(nn.Module):
    def __init__(self):
        super(OutputConvBlock_specularmap, self).__init__()
        # conv: c128(k1)
        self.cbr6 = ConvBNReluBlock(128, 128, 1, stride=1, padding=0)
        # conv: c64(k3)
        self.cbr7 = ConvBNReluBlock(128, 64, 3, stride=1, padding=1)
        # output: c*3(k1)
        self.out = nn.Conv2d(64, 2, 1, stride=1, padding=0)

    def forward(self, x):
        # deconv bilinear upsampling
        x = nn.functional.interpolate(
            x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.cbr6(x)
        x = self.cbr7(x)
        x = self.out(x)
        return x


class ResBlock(nn.Module):

    def __init__(self):
        super(ResBlock, self).__init__()
        self.bn = nn.BatchNorm2d(128, affine=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(128, 128, 3, padding=1, stride=1)
        self.bnr = nn.BatchNorm2d(128, affine=False)
        self.relur = nn.ReLU(inplace=True)
        self.convr = nn.Conv2d(128, 128, 3, padding=1, stride=1)

    def forward(self, x):
        residual = x
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.bnr(x)
        x = self.relur(x)
        x = self.convr(x)
        # add skip connection
        return residual + x


class StackedResBlocks(nn.Module):

    def __init__(self):
        super(StackedResBlocks, self).__init__()
        self.res1 = ResBlock()
        self.res2 = ResBlock()
        self.res3 = ResBlock()
        self.res4 = ResBlock()
        self.res5 = ResBlock()
        self.bn6r = nn.BatchNorm2d(128, affine=False)
        self.relu6r = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.bn6r(x)
        x = self.relu6r(x)
        return x

class EnvmapBlock(nn.Module):
    """
        Based on confidence-weighted average block. Confidence weighted average is not computed here.
        This block returns the blocks that contains the set of all envmap predictions (sort of like basis)
        and their corresponding confidence.
        returns (num_batches, 16, 16, 2048) feature block.
    """

    def __init__(self):
        super(EnvmapBlock, self).__init__()
        self.conv1 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(256, 512, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(512, 512, 3, stride=1, padding=1)

        self.groupnorm0 = nn.GroupNorm(32, 128)
        self.groupnorm1 = nn.GroupNorm(32, 256)
        self.groupnorm2 = nn.GroupNorm(32, 256)
        self.groupnorm3 = nn.GroupNorm(32, 512)
        self.groupnorm4 = nn.GroupNorm(32, 512)
        self.groupnorm5 = nn.GroupNorm(32, 512)

        self.prelu0 = torch.nn.PReLU(128)
        self.prelu1 = torch.nn.PReLU(256)
        self.prelu2 = torch.nn.PReLU(256)
        self.prelu3 = torch.nn.PReLU(512)
        self.prelu4 = torch.nn.PReLU(512)
        self.prelu5 = torch.nn.PReLU(512)

        self.out = nn.Conv2d(512, 512*4, kernel_size=3, stride=1, padding=1)

        # self.out = nn.Conv2d(512, 3, kernel_size =1, stride=(2,1), padding=8)
        # self.out = nn.ConvTranspose2d(512, 3, 3, stride=1, padding=0, output_padding=1)

    def forward(self, x):
        
        x = self.groupnorm0(x)
        x = self.prelu0(x)
        x = self.conv1(x)
        x = self.groupnorm1(x)
        x = self.prelu1(x)
        x = self.conv2(x)
        x = self.groupnorm2(x)
        x = self.prelu2(x)
        x = self.conv3(x)
        
        x = self.groupnorm3(x)
        x = self.prelu3(x)
        x = self.conv4(x)
        
        x = self.groupnorm4(x)
        x = self.prelu4(x)
        x = self.conv5(x)
        x = self.groupnorm5(x)
        x = self.prelu5(x)
        x = self.out(x)

        return x

#-------------------------I2I Transfer Model (UNet)----------------------------#


class Down2Conv1Maxpool(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    Padded -> no size change after convs.
    """

    def __init__(self, in_channels, out_channels):
        super(Down2Conv1Maxpool, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, groups=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1, bias=True, groups=1)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        before_pool = x
        x = self.maxpool(x)
        return x, before_pool


class Up2Conv1Transpose(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU activation follows each convolution.
    """

    def __init__(self, in_channels, out_channels):
        super(Up2Conv1Transpose, self).__init__()

        self.uptranspose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2)

        self.conv1 = nn.Conv2d(2*out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1, bias=True, groups=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1, bias=True, groups=1)

    def forward(self, x_from_encoder, x_from_decoder, envmap_fc2, i):
        """ Forward pass
        Arguments:
            x_from_encoder: tensor from the encoder pathway
            x_from_decoder: upconv'd tensor from the decoder pathway
            envmap_fc2: transformed environment features to be modulated with x_from_decoder
            i: index of upconv layer
        """
        x_from_decoder = self.uptranspose(x_from_decoder)

        if envmap_fc2.shape[0] != 0:
            # Setting up the dimensions
            envmap_fc2 = envmap_fc2.view([x_from_decoder.shape[0],envmap_fc2.shape[1],1,1])
            envmap_fc2 = envmap_fc2.expand(x_from_decoder.shape[0],envmap_fc2.shape[1],
                x_from_decoder.shape[2],x_from_decoder.shape[3])

            # print(" Layer %d x_from_decoder mean = %.5f and envmap_fc2 mean = %.5f " 
                            # % (i, torch.mean(x_from_decoder), torch.mean(envmap_fc2)))
            # Feature modulation: similar to StyleGAN's style modulation
            x_from_decoder = x_from_decoder + x_from_decoder * envmap_fc2 + envmap_fc2

        x = torch.cat((x_from_encoder, x_from_decoder), 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x


class UNetModel(nn.Module):
    """ 
    code ref: https://github.com/jaxony/unet-pytorch/blob/master/model.py
    `UNet` class is based on https://arxiv.org/abs/1505.04597
    The U-Net is a convolutional encoder-decoder neural network.
    Contextual spatial information (from the decoding,
    expansive pathway) about an input tensor is merged with
    information representing the localization of details
    (from the encoding, compressive pathway).
    Modifications to the original paper:
    (1) padding is used in 3x3 convolutions to prevent loss
        of border pixels
    (2) merging outputs does not require cropping due to (1)
    (3) residual connections can be used by specifying
        UNet(merge_mode='add')
    (4) if non-parametric upsampling is used in the decoder
        pathway (specified by upmode='upsample'), then an
        additional 1x1 2d convolution occurs after upsampling
        to reduce channel dimensionality by a factor of 2.
        This channel halving happens with the convolution in
        the tranpose convolution (specified by upmode='transpose')
    """

    def __init__(self, in_channels=6, out_channels=1, depth=5, num_filters1=64, cond_envmap="NONE", norm_type="Weber"):
        """
        Arguments:
            depth: int, number of MaxPools in the U-Net.
            start_filts: int, number of convolutional filters for the 
                first conv.
            up_mode: string, type of upconvolution. Choices: 'transpose'
                for transpose convolution or 'upsample' for nearest neighbour
                upsampling.
        """
        super(UNetModel, self).__init__()

        self.cond_envmap = cond_envmap
        self.norm_type = norm_type
        self.down_convs = []
        self.up_convs = []

        # create the encoder pathway and add to a list
        # down
        for i in range(depth-1):
            in_ch = in_channels if i == 0 else out_ch
            out_ch = num_filters1*(2**i)
            down_conv = Down2Conv1Maxpool(in_ch, out_ch)
            self.down_convs.append(down_conv)
        self.down_convs = nn.ModuleList(self.down_convs)

        # bottom
        # output
        in_ch = out_ch
        out_ch = num_filters1*(2**(i+1))
        self.bottom_conv1 = nn.Conv2d(
            in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True, groups=1)
        self.bottom_conv2 = nn.Conv2d(
            out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True, groups=1)

        # up
        for i in range(depth-1):
            in_ch = out_ch
            out_ch = in_ch // 2
            up_conv = Up2Conv1Transpose(in_ch, out_ch)
            self.up_convs.append(up_conv)
        self.up_convs = nn.ModuleList(self.up_convs)


        # Light feature modulation : depth
        self.lfm_depth = depth - 1
        # Light feature modulation: FC1
        self.LFM_fc1 = nn.Linear(1536, 128)
        # Light feature modulation: FC2
        self.LFM_fc2s = nn.ModuleList([
                            nn.Linear(128, 512 // pow(2,i)) for i in range(self.lfm_depth)
                        ])

        self.output_cov = nn.Conv2d(
            out_ch, out_channels, kernel_size=1, stride=1, groups=1)

    def forward(self, x, envmap):
        depth = len(self.down_convs)+1
        encoder_outs = []
        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)

        x = F.relu(self.bottom_conv1(x))
        x = F.relu(self.bottom_conv2(x))

        # If condition is provided, apply LFM
        if self.cond_envmap != "NONE":
            envmap = torch.flatten(envmap, start_dim=1)
            # Apply FC1, Relu only applied after FC1.
            envmap_fc1 = F.relu(self.LFM_fc1(envmap))
            # 

        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i+1)]
            """
                Light feature modulation : FC2
                gamma = FC(FC(envmap.reshape(1536,1), 128 x 1), x.shape[-1] x 1)
                x = x + gamma * x + beta
            """
            # Initialize
            envmap_fc2 = torch.empty(0)
            if self.cond_envmap != "NONE":
                # Apply FC2, Note: No Relu
                envmap_fc2 = self.LFM_fc2s[i](envmap_fc1)

            x = module(before_pool, x, envmap_fc2, i)

        x = self.output_cov(x)

        # No need to clamp for Weber normalization
        if self.norm_type == "Naive":
            x = torch.clamp_min(x, 0)

        return x

#---------------------------The Decomposition Model----------------------------#


class DecompModel(nn.Module):

    def __init__(self, cond_envmap="ORG"):
        super(DecompModel, self).__init__()

        self.cond_envmap = cond_envmap
        # 3 input convs
        # conv1: c64(k7)
        self.cbr1 = ConvBNReluBlock(3, 64, 7, stride=1, padding=3)
        # conv2: c128(k3)
        self.cbr2 = ConvBNReluBlock(64, 128, 3, stride=1, padding=1)
        # conv3: c*128(k3)
        self.conv3 = nn.Conv2d(128, 128, 3, stride=2, padding=1)

        # normal
        # 5 residual blocks + bn + relu
        self.nresblocks = StackedResBlocks()
        # output convs
        self.noutconvs = OutputConvBlock()

        # albedo
        # 5 residual blocks + bn + relu
        self.aresblocks = StackedResBlocks()
        # output convs
        self.aoutconvs = OutputConvBlock()

        self.sresblocks = StackedResBlocks()
        # output convs
        self.soutconvs = OutputConvBlock_specularmap()

        # Envmap
        self.envmapblocks = EnvmapBlock()

        self.softmax = nn.Softmax()
        self.softplus = nn.Softplus()

        self.WEIGHTED_POOLING = False

    def forward(self, input_img, mask):
        # 3 input convs
        encoded_img = self.cbr1(input_img)
        encoded_img = self.cbr2(encoded_img)
        encoded_img = self.conv3(encoded_img)

        # normal resblocks + bn + rl
        normal = self.nresblocks(encoded_img)
        # albedo resblocks + bn + rl
        albedo = self.aresblocks(encoded_img)
        # specular resblocks + bn + rl

        # normal output conv
        normal = self.noutconvs(normal)
        # albedo output conv
        albedo = self.aoutconvs(albedo)

        roughness_rho = self.sresblocks(encoded_img)

        roughness_rho = self.soutconvs(roughness_rho)
        roughness = torch.index_select(
            roughness_rho, dim=1, index=torch.tensor([0]).cuda())
        rho = torch.index_select(
            roughness_rho, dim=1, index=torch.tensor([1]).cuda())

        # Mask out background
        # normal = torch.clamp(normal*mask, 0)
        # inverted_mask = torch.where(mask > 0, torch.tensor([0.0]).cuda(), torch.tensor([1.0]).cuda())
        normal = torch.where(mask > 0, normal, mask*torch.min(normal))
        # normal[mask == 0] = torch.min(normal)
        # shading[mask == 0] = torch.min(shading)
        # albedo[mask == 0] = torch.min(albedo)
        # print(torch.min(normal), torch.max(normal))

        if self.cond_envmap == "ORG" or self.cond_envmap == "SH":
            # Call Envmap Block and returns (num_batches, 16, 16, 2048) feature
            basis = self.envmapblocks(encoded_img)
            basis = torch.reshape(basis, (basis.size()[0],4,16,32,-1))
            envmaps = basis[:,:3,:,:,:]
            confidences = basis[:,3:4,:,:,:]

            w, h, c = map(int, confidences.size()[2:])
            confidences = torch.reshape(confidences, shape=[-1, w * h, c])
            confidences = self.softplus(confidences) #softplus activation?
            confidences = torch.reshape(confidences, shape=[-1, 1, w, h, c])

            envmap_w_confidence = envmaps * confidences
            envmap_w_confidence = torch.sum(envmap_w_confidence, dim=4)

        elif self.cond_envmap == "NONE":
            envmap_w_confidence = torch.zeros((1,3,16,32))
            confidences = torch.zeros((1,1,16,32,1))

        # if not self.WEIGHTED_POOLING:
        # #     # Simply average pooling
        # #     # envmap_weighted = tf.nn.l2_normalize(envmap_weighted, 3)
        #     envmap_weighted = nn.functional.normalize(envmap_weighted,dim=3,p=2)

        # else:
        #     pass
            # Pooling using a simple FC, as noted in the paper
            # fc2 = tf.nn.l2_normalize(
            #   slim.conv2d(
            #       tf.nn.relu(fc2),
            #       3, [15, 15],
            #       scope='fc_pooling',
            #       activation_fn=None,
            #       padding='VALID'), 3)

        return  albedo, normal, roughness, rho, envmap_w_confidence, confidences[:,:,:,:,0]

