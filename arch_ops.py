# Copyright 2023 Nikolai Körber. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
This file contains

1) a partial port of compressai.layers
    - conv3x3
    - ResidualBlock
    - ResidualBlockWithStride
    - ResidualBlockUpsample
    - subpel_conv3x3
    - AttentionBlock

    Source: https://interdigitalinc.github.io/CompressAI/_modules/compressai/layers/layers.html.

2) a partial port of tcm
    - conv
    - conv1x1
    - ConvTransBlock
    - SWAtten

    Source: https://github.com/jmliu206/LIC_TCM/blob/main/models/tcm.py

We keep the original header definition for better comparability (although the in_ch argument is not required in TF).
"""

import sys

# support both local environment + google colab (update as required)
sys.path.append('swin-transformers-tf')
sys.path.append('/content/LIC-TCM/swin-transformers-tf')

import tensorflow as tf
import tensorflow_compression as tfc

from tensorflow.keras.layers import Conv2D, Concatenate
from swins.blocks.swin_transformer_block import SwinTransformerBlock as Block
from swins.blocks.stage_block import BasicLayer as SwinBlock


def conv(out_channels, kernel_size=5, stride=2):
    return tf.keras.layers.Conv2D(
        filters=out_channels,
        kernel_size=kernel_size,
        strides=stride,
        padding='same',
    )


def conv1x1(in_ch, out_ch, stride=1):
    """1x1 convolution."""
    return tf.keras.layers.Conv2D(out_ch, kernel_size=1, strides=stride)


def conv3x3(in_ch, out_ch, stride=1):
    """3x3 convolution with padding."""
    return tf.keras.layers.Conv2D(out_ch, kernel_size=3, strides=stride, padding='same')


class ResidualBlock(tf.keras.layers.Layer):
    """Simple residual block with two 3x3 convolutions.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
    """

    def __init__(self, in_ch, out_ch):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_ch, out_ch)
        self.leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.01)
        self.conv2 = conv3x3(out_ch, out_ch)
        if in_ch != out_ch:
            self.skip = conv1x1(in_ch, out_ch)
        else:
            self.skip = None

    def call(self, x):
        identity = x

        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.leaky_relu(out)

        if self.skip is not None:
            identity = self.skip(x)

        out = out + identity
        return out


class ResidualBlockWithStride(tf.keras.layers.Layer):
    """Residual block with a stride on the first convolution.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        stride (int): stride value (default: 2)
    """

    def __init__(self, in_ch, out_ch, stride=2):
        super(ResidualBlockWithStride, self).__init__()
        self.conv1 = conv3x3(in_ch, out_ch, stride=stride)
        self.leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.01)
        self.conv2 = conv3x3(out_ch, out_ch)
        self.gdn = tfc.GDN()  # removed argument
        if stride != 1 or in_ch != out_ch:
            self.skip = conv1x1(in_ch, out_ch, stride=stride)
        else:
            self.skip = None

    def call(self, x):
        identity = x
        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.gdn(out)

        if self.skip is not None:
            identity = self.skip(x)

        out += identity
        return out


class ResidualBlockUpsample(tf.keras.layers.Layer):
    """Residual block with sub-pixel upsampling on the last convolution.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        upsample (int): upsampling factor (default: 2)
    """

    def __init__(self, in_ch, out_ch, upsample=2):
        super(ResidualBlockUpsample, self).__init__()
        self.subpel_conv = subpel_conv3x3(in_ch, out_ch, upsample)
        self.leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.01)
        self.conv = conv3x3(out_ch, out_ch)
        self.igdn = tfc.GDN(inverse=True)
        self.upsample = subpel_conv3x3(in_ch, out_ch, upsample)

    def call(self, x):
        identity = x
        out = self.subpel_conv(x)
        out = self.leaky_relu(out)
        out = self.conv(out)
        out = self.igdn(out)
        identity = self.upsample(x)
        out += identity
        return out


def subpel_conv3x3(in_ch, out_ch, r=1):
    """3x3 sub-pixel convolution for up-sampling."""
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(out_ch * r ** 2, kernel_size=3, padding='same'),
        tf.keras.layers.Lambda(lambda x: tf.nn.depth_to_space(x, r))
    ])


class ConvTransBlock(tf.keras.layers.Layer):
    """Translated from https://github.com/jmliu206/LIC_TCM/blob/main/models/tcm.py#L238"""

    def __init__(self, conv_dim, trans_dim, head_dim, window_size, drop_path, type_w='W'):
        super(ConvTransBlock, self).__init__()
        self.conv_dim = conv_dim
        self.trans_dim = trans_dim
        self.head_dim = head_dim
        self.window_size = window_size
        self.drop_path = drop_path
        self.type_w = type_w
        assert self.type_w in ['W', 'SW']
        shift_size = 0
        if self.type_w == 'SW':
            shift_size = window_size // 2
        self.trans_block = Block(dim=self.trans_dim, num_heads=trans_dim // head_dim, head_dim=self.head_dim,
                                 window_size=self.window_size, shift_size=shift_size)

        self.conv1_1 = Conv2D(self.conv_dim + self.trans_dim, 1, 1, padding='valid', use_bias=True)
        self.conv1_2 = Conv2D(self.conv_dim + self.trans_dim, 1, 1, padding='valid', use_bias=True)

        self.conv_block = ResidualBlock(self.conv_dim, self.conv_dim)

    def call(self, x):
        conv_x, trans_x = tf.split(self.conv1_1(x), (self.conv_dim, self.trans_dim), axis=-1)
        conv_x = self.conv_block(conv_x) + conv_x
        trans_x = self.trans_block(trans_x)
        res = self.conv1_2(Concatenate(axis=-1)([conv_x, trans_x]))
        x = x + res
        return x


class AttentionBlock(tf.keras.layers.Layer):
    """Self attention block.

    Simplified variant from `"Learned Image Compression with
    Discretized Gaussian Mixture Likelihoods and Attention Modules"
    <https://arxiv.org/abs/2001.01568>`_, by Zhengxue Cheng, Heming Sun, Masaru
    Takeuchi, Jiro Katto.

    Args:
        N (int): Number of channels)
    """

    def __init__(self, N: int):
        super(AttentionBlock, self).__init__()

        class ResidualUnit(tf.keras.layers.Layer):
            """Simple residual unit."""

            def __init__(self):
                super(ResidualUnit, self).__init__()
                self.conv = tf.keras.Sequential([
                    conv1x1(N, N // 2),
                    tf.keras.layers.ReLU(),
                    conv3x3(N // 2, N // 2),
                    tf.keras.layers.ReLU(),
                    conv1x1(N // 2, N)
                ])
                self.relu = tf.keras.layers.ReLU()

            def call(self, x: tf.Tensor) -> tf.Tensor:
                identity = x
                out = self.conv(x)
                out += identity
                out = self.relu(out)
                return out

        self.conv_a = tf.keras.Sequential([ResidualUnit() for _ in range(3)])

        self.conv_b = tf.keras.Sequential([
                                              ResidualUnit() for _ in range(3)
                                          ] + [conv1x1(N, N)])

    def call(self, x: tf.Tensor) -> tf.Tensor:
        identity = x
        a = self.conv_a(x)
        b = self.conv_b(x)
        out = a * tf.sigmoid(b)
        out += identity
        return out


class SWAtten(AttentionBlock):
    """Translated from https://github.com/jmliu206/LIC_TCM/blob/main/models/tcm.py#L266"""

    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path, inter_dim=192):
        super(SWAtten, self).__init__(N=inter_dim)
        self.non_local_block = SwinBlock(dim=inter_dim, out_dim=inter_dim, depth=2, num_heads=inter_dim // head_dim,
                                         head_dim=head_dim, window_size=window_size, drop_path=drop_path)

        self.in_conv = conv1x1(input_dim, inter_dim)
        self.out_conv = conv1x1(inter_dim, output_dim)

    def call(self, x):
        x = self.in_conv(x)
        identity = x
        z = self.non_local_block(x)
        a = self.conv_a(x)
        b = self.conv_b(z)
        out = a * tf.sigmoid(b)
        out += identity
        out = self.out_conv(out)
        return out
