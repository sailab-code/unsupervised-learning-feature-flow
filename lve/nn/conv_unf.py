import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2dUnf(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros', uniform_init=None):
        self.uniform_init = uniform_init
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.vol = self.weight.shape[1] * self.weight.shape[2] * self.weight.shape[3]

        if groups != 1:
            raise ValueError('Only groups=1 is supported (groups=' + str(groups) + ')')
        if padding_mode != 'zeros':
            raise ValueError('Only padding_mode=zeros is supported (padding_mode=' + str(padding_mode) + ')')

    def reset_parameters(self):
        if self.uniform_init is not None:
            nn.init.uniform_(self.weight, -self.uniform_init, self.uniform_init)
        else:
            super().reset_parameters()

    def forward(self, input, patches_needed=True, patch_row_col=None):

        if patches_needed:
            if patch_row_col is None:

                # patches are: batch_size x self.vol x wh
                patches = F.unfold(input, self.kernel_size, stride=self.stride,
                                   padding=self.padding, dilation=self.dilation).view(input.shape[0], self.vol,
                                                                                      input.shape[2] * input.shape[3])

                conv = torch.bmm(
                    self.weight.view(self.out_channels, self.vol).expand(input.shape[0], -1, -1), patches) \
                    .view(input.shape[0], self.out_channels, input.shape[2], input.shape[3])

                if self.bias is None:
                    return conv, patches
                else:
                    return conv + self.bias.view(1, self.out_channels, 1, 1), patches  # broadcasting bias addition
            else:

                # here we perform a convolution using class tools from pytorch, and be gather the tubelet of patches in
                # the trajectory of the focus of attention over the given input batch
                kernel_size = self.kernel_size[0]
                k = kernel_size // 2
                b = patch_row_col.shape[0]
                device = patch_row_col.device

                padded_input = F.pad(input, pad=(k, k, k, k, 0, 0, 0, 0))

                assert b == padded_input.shape[0]
                assert self.kernel_size[0] == self.kernel_size[1]

                c = padded_input.shape[1]
                h = padded_input.shape[2]
                w = padded_input.shape[3]

                padded_input = padded_input.view(b, c, h * w)

                x_cord = torch.arange(kernel_size, device=device).repeat(kernel_size).view(kernel_size, kernel_size)
                y_cord = x_cord.t().contiguous()
                x_cord = x_cord.view(-1).unsqueeze(0).repeat(b, 1) + patch_row_col[:, 1]
                y_cord = y_cord.view(-1).unsqueeze(0).repeat(b, 1) + patch_row_col[:, 0]
                indices = (y_cord * w + x_cord).view(b, 1, kernel_size * kernel_size).repeat(1, c, 1)
                patches = torch.gather(padded_input, 2, indices).view(b, self.vol, 1)  # batch_size x self.vol x 1

                return super().forward(input), patches  # classic convolution + single patches
        else:
            return super().forward(input), None  # classic convolution
