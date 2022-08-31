# Copyright (c) 2021 Mobvoi Inc (Binbin Zhang, Di Wu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified from ESPnet(https://github.com/espnet/espnet)


"""Subsampling layer definition."""

from typing import Tuple, Union

import torch


class BaseSubsampling(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.right_context = 0
        self.subsampling_rate = 1

    def position_encoding(self, offset: Union[int, torch.Tensor],
                          size: int) -> torch.Tensor:
        return self.pos_enc.position_encoding(offset, size)


class Conv2dSubsampling4(BaseSubsampling):
    """Convolutional 2D subsampling (to 1/4 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.

    """
    def __init__(self, idim: int, odim: int, dropout_rate: float,
                 pos_enc_class: torch.nn.Module):
        """Construct an Conv2dSubsampling4 object."""
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim))
        self.pos_enc = pos_enc_class
        # The right context for every conv layer is computed by:
        # (kernel_size - 1) * frame_rate_of_this_layer
        self.subsampling_rate = 4
        # 6 = (3 - 1) * 1 + (3 - 1) * 2
        self.right_context = 6

    def forward(
            self,
            x: torch.Tensor,
            offset: Union[int, torch.Tensor] = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    #) -> Tuple[torch.Tensor]:
            #x_mask: torch.Tensor, #部署阶段转mnn，将不需要的参数都略去

        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 4.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 4.
            torch.Tensor: positional encoding

        """
        #x = x.unsqueeze(1)  # (b, c=1, t, f)
        x = self.conv(x)
        #b, c, t, f = x.size()   # 1, c=512, t=16 chunksize, f=19
        #x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        
        #x = self.out(x.transpose(1, 2).contiguous().view(1, 16, -1))    #部署阶段， batch=1，t=16为chunksize大小 , out为全连接层
        x = self.out(x.transpose(1, 2).reshape(1, 16, -1))    #部署阶段， batch=1，t=16为chunksize大小 , out为全连接层
        
        x, pos_emb = self.pos_enc(x, offset)  #self.pos_enc对x进行缩放22倍，pos为只根据offset和size生成额位置编码
        #x = x * 22.62 #self.xscale, math.sqrt(512)==22.627
        #x = x * self.xscale  #, math.sqrt(512)==22.627
        return x, pos_emb #, x_mask[:, :, :-2:2][:, :, :-2:2]
        #return x
