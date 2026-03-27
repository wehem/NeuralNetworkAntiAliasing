"""
convert.py - Convert a Keras NNAA model (improved detail‑restoration version) to ReShade FX compute shader.

This script loads the trained model with residual blocks, batch norm folding, and upsampling + conv,
and generates the equivalent HLSL compute shader code.

Architecture (after folding BN into conv):
  - Detail branch: 2x Conv2D(32,3) + PReLU
  - Context branch: Conv2D(32,8,stride=2) + PReLU
                   3x residual blocks (each: Conv2D(32,3)+PReLU, Conv2D(32,3)+ skip + PReLU)
                   UpSampling2D + Conv2D(32,3) + PReLU
  - Fusion: Concatenate + Conv2D(32,3) + PReLU + Conv2D(1,1) -> residual, then upscale to full-res

Usage:
  python convert.py [model_path] [output_path]
"""

import sys
import numpy as np
import tensorflow as tf

def fold_batch_norm(conv_weights, conv_bias, bn_weights):
    """
    Fold batch normalization into convolution weights.
    bn_weights = [gamma, beta, moving_mean, moving_variance]
    Returns (new_weights, new_bias)
    """
    gamma, beta, mean, var = bn_weights
    epsilon = 1e-5  # typical keras default
    scale = gamma / np.sqrt(var + epsilon)
    new_weights = conv_weights * scale.reshape(1, 1, 1, -1)
    if conv_bias is not None:
        new_bias = beta + scale * (conv_bias - mean)
    else:
        new_bias = beta - scale * mean
    return new_weights, new_bias

def fmt(v):
    """Format a float value as a string, matching the original shader precision."""
    return repr(float(v))

def generate_header():
    """Generate the license, includes, defines, and texture/storage declarations."""
    return """/**
 * MIT License
 * 
 * Copyright (c) 2025 Leo Calvis
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "ReShadeUI.fxh"
#include "ReShade.fxh"

#define K_SIZE 128

// Half‑res luma (packed 2x2)
texture2D texLuma_nnaa1
{
    Width = BUFFER_WIDTH / 2 + 1;
    Height = BUFFER_HEIGHT / 2 + 1;
    MipLevels = 0;
    Format = RGBA16F;
};
storage2D storageLuma_nnaa1
{
    Texture = texLuma_nnaa1;
    MipLevel = 0;
};

// Detail branch output (half‑res, 32 channels stored in 8 rows of RGBA)
texture2D texDetail_nnaa1
{
    Width = (BUFFER_WIDTH / 2);
    Height = (BUFFER_HEIGHT / 2) * 8;
    MipLevels = 0;
    Format = RGBA16F;
};
storage2D storageDetail_nnaa1
{
    Texture = texDetail_nnaa1;
    MipLevel = 0;
};

// Context branch intermediate textures (quarter‑res, 32 channels)
texture2D texContext0_nnaa1
{
    Width = (BUFFER_WIDTH / 4);
    Height = (BUFFER_HEIGHT / 4) * 8;
    MipLevels = 0;
    Format = RGBA16F;
};
storage2D storageContext0_nnaa1
{
    Texture = texContext0_nnaa1;
    MipLevel = 0;
};

texture2D texContext1_nnaa1
{
    Width = (BUFFER_WIDTH / 4);
    Height = (BUFFER_HEIGHT / 4) * 8;
    MipLevels = 0;
    Format = RGBA16F;
};
storage2D storageContext1_nnaa1
{
    Texture = texContext1_nnaa1;
    MipLevel = 0;
};

texture2D texContext2_nnaa1
{
    Width = (BUFFER_WIDTH / 4);
    Height = (BUFFER_HEIGHT / 4) * 8;
    MipLevels = 0;
    Format = RGBA16F;
};
storage2D storageContext2_nnaa1
{
    Texture = texContext2_nnaa1;
    MipLevel = 0;
};

// Upsampled context output (half‑res, 32 channels)
texture2D texContextUp_nnaa1
{
    Width = (BUFFER_WIDTH / 2);
    Height = (BUFFER_HEIGHT / 2) * 8;
    MipLevels = 0;
    Format = RGBA16F;
};
storage2D storageContextUp_nnaa1
{
    Texture = texContextUp_nnaa1;
    MipLevel = 0;
};

// Final residual (full‑res)
texture2D texResult_nnaa1
{
    Width = BUFFER_WIDTH;
    Height = BUFFER_HEIGHT;
    MipLevels = 0;
    Format = R16F;
};
storage2D storageResult_nnaa1
{
    Texture = texResult_nnaa1;
    MipLevel = 0;
};

sampler2D samplerResult_nnaa1
{
    Texture = texResult_nnaa1;
};

[shader("compute")]
void GetLuma(int3 id : SV_DispatchThreadID)
{
    min16float4 luma = float4(
        dot(tex2Dfetch(ReShade::BackBuffer, id.xy * int2(2, 2) + int2(-1, -1)).rgb, min16float3(0.299, 0.587, 0.114)),
        dot(tex2Dfetch(ReShade::BackBuffer, id.xy * int2(2, 2) + int2(0, -1)).rgb, min16float3(0.299, 0.587, 0.114)),
        dot(tex2Dfetch(ReShade::BackBuffer, id.xy * int2(2, 2) + int2(-1, 0)).rgb, min16float3(0.299, 0.587, 0.114)),
        dot(tex2Dfetch(ReShade::BackBuffer, id.xy * int2(2, 2) + int2(0, 0)).rgb, min16float3(0.299, 0.587, 0.114)));
    tex2Dstore(storageLuma_nnaa1, id.xy, luma);
}

[shader("pixel")]
float4 ApplyNN(float4 pos : SV_Position) : SV_Target
{
    min16float luma = tex2Dfetch(samplerResult_nnaa1, pos.xy).r;
    min16float4 old_color = tex2Dfetch(ReShade::BackBuffer, pos.xy);
    min16float y = dot(old_color.rgb, min16float3(0.299, 0.587, 0.114)) + luma;
    min16float cb = dot(old_color.rgb, min16float3(-0.1687, -0.3313, 0.5));
    min16float cr = dot(old_color.rgb, min16float3(0.5, -0.4187, -0.0813));
    return float4(y + 1.402 * cr, y - 0.34414 * cb - 0.71414 * cr, y + 1.772 * cb, old_color.a);
}

"""

def generate_detail_conv_1(conv_weights, conv_bias, prelu_alpha):
    """First detail conv (3x3, stride=1, half‑res luma) -> 32 channels."""
    lines = []
    lines.append('[shader("compute")]')
    lines.append('void Layer_detail_1(int3 id : SV_DispatchThreadID)')
    lines.append('{')
    lines.append('    if(id.x >= (BUFFER_WIDTH / 2)) return;')
    num_out = 32
    num_groups = 8
    for g in range(num_groups):
        b = conv_bias[g*4:(g+1)*4]
        lines.append(f'    min16float4 f_out_{g} = min16float4({fmt(b[0])},{fmt(b[1])},{fmt(b[2])},{fmt(b[3])});')
    lines.append('    min16float4 f_in;')
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            lines.append(f'    f_in = tex2Dfetch(storageLuma_nnaa1, id.xy + int2({dx}, {dy}));')
            for c in range(4):
                channel_name = ['x', 'y', 'z', 'w'][c]
                in_ch = c
                ky = dy + 1
                kx = dx + 1
                for out_g in range(num_groups):
                    w = conv_weights[ky, kx, in_ch, out_g*4:(out_g+1)*4]
                    lines.append(f'    f_out_{out_g} += min16float4({fmt(w[0])},{fmt(w[1])},{fmt(w[2])},{fmt(w[3])}) * f_in.{channel_name};')
    alpha = prelu_alpha.reshape(-1)
    # store to temporary texture? Actually we need to keep intermediate for second detail conv.
    # We'll store to an intermediate texture (half‑res, 32 channels) but we already have storageDetail_nnaa1.
    # We'll use storageDetail_nnaa1 for the output of the first detail conv? The second detail conv will read from there.
    # Let's store to storageDetail_nnaa1 after first conv, then second conv reads from it and writes back to same texture? That would require ping-pong.
    # Simpler: use two separate textures: storageDetail0_nnaa1 and storageDetail1_nnaa1.
    # But to keep texture count minimal, we can store first conv output in storageDetail_nnaa1, then second conv reads from it and writes to storageDetail_nnaa1 (overwrites). That's fine because we don't need the first after second.
    for g in range(num_groups):
        for c_idx, c_name in enumerate(['x', 'y', 'z', 'w']):
            ch = g*4 + c_idx
            lines.append(f'    if(f_out_{g}.{c_name} < 0) f_out_{g}.{c_name} *= {fmt(alpha[ch])};')
        lines.append(f'    tex2Dstore(storageDetail_nnaa1, id.xy * int2(1, 8) + int2(0, {g}), f_out_{g});')
    lines.append('}')
    return '\n'.join(lines)

def generate_detail_conv_2(conv_weights, conv_bias, prelu_alpha):
    """Second detail conv (3x3, stride=1) reading from storageDetail_nnaa1 (32ch) -> 32ch."""
    lines = []
    lines.append('[shader("compute")]')
    lines.append('void Layer_detail_2(int3 id : SV_DispatchThreadID)')
    lines.append('{')
    lines.append('    if(id.x >= (BUFFER_WIDTH / 2)) return;')
    num_out = 32
    num_groups = 8
    for g in range(num_groups):
        b = conv_bias[g*4:(g+1)*4]
        lines.append(f'    min16float4 f_out_{g} = min16float4({fmt(b[0])},{fmt(b[1])},{fmt(b[2])},{fmt(b[3])});')
    lines.append('    min16float4 f_in;')
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            for in_g in range(num_groups):
                lines.append(f'    f_in = tex2Dfetch(storageDetail_nnaa1, id.xy * int2(1, 8) + int2({dx}, {in_g}));')
                for c in range(4):
                    channel_name = ['x', 'y', 'z', 'w'][c]
                    in_ch = in_g*4 + c
                    ky = dy + 1
                    kx = dx + 1
                    for out_g in range(num_groups):
                        w = conv_weights[ky, kx, in_ch, out_g*4:(out_g+1)*4]
                        lines.append(f'    f_out_{out_g} += min16float4({fmt(w[0])},{fmt(w[1])},{fmt(w[2])},{fmt(w[3])}) * f_in.{channel_name};')
    alpha = prelu_alpha.reshape(-1)
    for g in range(num_groups):
        for c_idx, c_name in enumerate(['x', 'y', 'z', 'w']):
            ch = g*4 + c_idx
            lines.append(f'    if(f_out_{g}.{c_name} < 0) f_out_{g}.{c_name} *= {fmt(alpha[ch])};')
        lines.append(f'    tex2Dstore(storageDetail_nnaa1, id.xy * int2(1, 8) + int2(0, {g}), f_out_{g});')
    lines.append('}')
    return '\n'.join(lines)

def generate_context_conv0(conv_weights, conv_bias, prelu_alpha):
    """Context branch first conv (8x8, stride=2) from half‑res luma to quarter‑res 32ch."""
    lines = []
    lines.append('[shader("compute")]')
    lines.append('void Layer_context0(int3 id : SV_DispatchThreadID)')
    lines.append('{')
    lines.append('    if(id.x >= (BUFFER_WIDTH / 4)) return;')
    num_out = 32
    num_groups = 8
    for g in range(num_groups):
        b = conv_bias[g*4:(g+1)*4]
        lines.append(f'    min16float4 f_out_{g} = min16float4({fmt(b[0])},{fmt(b[1])},{fmt(b[2])},{fmt(b[3])});')
    lines.append('    min16float4 f_in;')
    for dy in range(-1, 3):
        for dx in range(-1, 3):
            lines.append(f'    f_in = tex2Dfetch(storageLuma_nnaa1, id.xy + int2({dx}, {dy}));')
            for c in range(4):
                channel_name = ['x', 'y', 'z', 'w'][c]
                kx = dx*2 + 2 + (c % 2)
                ky = dy*2 + 2 + (c // 2)
                for out_g in range(num_groups):
                    w = conv_weights[ky, kx, 0, out_g*4:(out_g+1)*4]
                    lines.append(f'    f_out_{out_g} += min16float4({fmt(w[0])},{fmt(w[1])},{fmt(w[2])},{fmt(w[3])}) * f_in.{channel_name};')
    alpha = prelu_alpha.reshape(-1)
    for g in range(num_groups):
        for c_idx, c_name in enumerate(['x', 'y', 'z', 'w']):
            ch = g*4 + c_idx
            lines.append(f'    if(f_out_{g}.{c_name} < 0) f_out_{g}.{c_name} *= {fmt(alpha[ch])};')
        lines.append(f'    tex2Dstore(storageContext0_nnaa1, id.xy * int2(1, 8) + int2(0, {g}), f_out_{g});')
    lines.append('}')
    return '\n'.join(lines)

def generate_residual_block(block_idx, conv1_weights, conv1_bias, prelu1_alpha,
                            conv2_weights, conv2_bias, prelu2_alpha,
                            src_storage, dst_storage):
    """Generate a compute shader for one residual block (two convs + skip addition)."""
    lines = []
    lines.append(f'[shader("compute")]')
    lines.append(f'void Layer_resblock_{block_idx}(int3 id : SV_DispatchThreadID)')
    lines.append('{')
    lines.append('    if(id.x >= (BUFFER_WIDTH / 4)) return;')
    num_groups = 8
    # first conv
    for g in range(num_groups):
        b = conv1_bias[g*4:(g+1)*4]
        lines.append(f'    min16float4 f1_out_{g} = min16float4({fmt(b[0])},{fmt(b[1])},{fmt(b[2])},{fmt(b[3])});')
    lines.append('    min16float4 f_in;')
    # 3x3 conv1
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            for in_g in range(num_groups):
                lines.append(f'    f_in = tex2Dfetch({src_storage}, id.xy * int2(1, 8) + int2({dx}, {in_g}));')
                for c in range(4):
                    channel_name = ['x', 'y', 'z', 'w'][c]
                    in_ch = in_g*4 + c
                    ky = dy + 1
                    kx = dx + 1
                    for out_g in range(num_groups):
                        w = conv1_weights[ky, kx, in_ch, out_g*4:(out_g+1)*4]
                        lines.append(f'    f1_out_{out_g} += min16float4({fmt(w[0])},{fmt(w[1])},{fmt(w[2])},{fmt(w[3])}) * f_in.{channel_name};')
    # prelu1
    alpha1 = prelu1_alpha.reshape(-1)
    for g in range(num_groups):
        for c_idx, c_name in enumerate(['x', 'y', 'z', 'w']):
            ch = g*4 + c_idx
            lines.append(f'    if(f1_out_{g}.{c_name} < 0) f1_out_{g}.{c_name} *= {fmt(alpha1[ch])};')
    # second conv
    for g in range(num_groups):
        b = conv2_bias[g*4:(g+1)*4]
        lines.append(f'    min16float4 f2_out_{g} = min16float4({fmt(b[0])},{fmt(b[1])},{fmt(b[2])},{fmt(b[3])});')
    # 3x3 conv2
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            for in_g in range(num_groups):
                lines.append(f'    f_in = f1_out_{in_g};')
                for c in range(4):
                    channel_name = ['x', 'y', 'z', 'w'][c]
                    in_ch = in_g*4 + c
                    ky = dy + 1
                    kx = dx + 1
                    for out_g in range(num_groups):
                        w = conv2_weights[ky, kx, in_ch, out_g*4:(out_g+1)*4]
                        lines.append(f'    f2_out_{out_g} += min16float4({fmt(w[0])},{fmt(w[1])},{fmt(w[2])},{fmt(w[3])}) * f1_out_{in_g}.{channel_name};')
    # add skip (input texture)
    for g in range(num_groups):
        lines.append(f'    f_in = tex2Dfetch({src_storage}, id.xy * int2(1, 8) + int2(0, {g}));')
        lines.append(f'    f2_out_{g} += f_in;')
    # prelu2
    alpha2 = prelu2_alpha.reshape(-1)
    for g in range(num_groups):
        for c_idx, c_name in enumerate(['x', 'y', 'z', 'w']):
            ch = g*4 + c_idx
            lines.append(f'    if(f2_out_{g}.{c_name} < 0) f2_out_{g}.{c_name} *= {fmt(alpha2[ch])};')
        lines.append(f'    tex2Dstore({dst_storage}, id.xy * int2(1, 8) + int2(0, {g}), f2_out_{g});')
    lines.append('}')
    return '\n'.join(lines)

def generate_upsample_conv(conv_weights, conv_bias, prelu_alpha):
    """Upsample + 3x3 conv: from quarter‑res (storageContext2_nnaa1) to half‑res (storageContextUp_nnaa1)."""
    lines = []
    lines.append('[shader("compute")]')
    lines.append('void Layer_upsample_conv(int3 id : SV_DispatchThreadID)')
    lines.append('{')
    lines.append('    if(id.x >= (BUFFER_WIDTH / 2)) return;')
    num_groups = 8
    for g in range(num_groups):
        b = conv_bias[g*4:(g+1)*4]
        lines.append(f'    min16float4 f_out_{g} = min16float4({fmt(b[0])},{fmt(b[1])},{fmt(b[2])},{fmt(b[3])});')
    lines.append('    min16float4 f_in;')
    # nearest neighbor upsampling: each half‑res pixel corresponds to a quarter‑res pixel at (id.xy // 2)
    lines.append('    uint2 qcoord = id.xy / 2;')
    # fetch 3x3 neighborhood from quarter‑res
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            lines.append(f'    uint2 c = qcoord + int2({dx}, {dy});')
            lines.append('    if(c.x < (BUFFER_WIDTH / 4) && c.y < (BUFFER_HEIGHT / 4)) {')
            for in_g in range(num_groups):
                lines.append(f'        f_in = tex2Dfetch(storageContext2_nnaa1, c * int2(1, 8) + int2(0, {in_g}));')
                for c_idx in range(4):
                    channel_name = ['x', 'y', 'z', 'w'][c_idx]
                    in_ch = in_g*4 + c_idx
                    ky = dy + 1
                    kx = dx + 1
                    for out_g in range(num_groups):
                        w = conv_weights[ky, kx, in_ch, out_g*4:(out_g+1)*4]
                        lines.append(f'        f_out_{out_g} += min16float4({fmt(w[0])},{fmt(w[1])},{fmt(w[2])},{fmt(w[3])}) * f_in.{channel_name};')
            lines.append('    }')
    # prelu
    alpha = prelu_alpha.reshape(-1)
    for g in range(num_groups):
        for c_idx, c_name in enumerate(['x', 'y', 'z', 'w']):
            ch = g*4 + c_idx
            lines.append(f'    if(f_out_{g}.{c_name} < 0) f_out_{g}.{c_name} *= {fmt(alpha[ch])};')
        lines.append(f'    tex2Dstore(storageContextUp_nnaa1, id.xy * int2(1, 8) + int2(0, {g}), f_out_{g});')
    lines.append('}')
    return '\n'.join(lines)

def generate_fusion(conv3_weights, conv3_bias, prelu3_alpha, final_conv_weights, final_conv_bias):
    """Fusion: read detail (storageDetail_nnaa1) and context up (storageContextUp_nnaa1), apply 3x3 conv, then 1x1 conv, then upscale to full-res."""
    lines = []
    lines.append('[shader("compute")]')
    lines.append('void Layer_fusion(int3 id : SV_DispatchThreadID)')
    lines.append('{')
    lines.append('    if(id.x >= (BUFFER_WIDTH / 2)) return;')
    num_groups = 8
    # First 3x3 conv (32 output channels)
    for g in range(num_groups):
        b = conv3_bias[g*4:(g+1)*4]
        lines.append(f'    min16float4 f3_out_{g} = min16float4({fmt(b[0])},{fmt(b[1])},{fmt(b[2])},{fmt(b[3])});')
    lines.append('    min16float4 f_in;')
    # 3x3 convolution on concatenated input (64 channels: 32 from detail, 32 from context up)
    # For each spatial offset, we need to fetch both detail and context up.
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            # detail part (channels 0-31)
            for in_g in range(num_groups):
                lines.append(f'    f_in = tex2Dfetch(storageDetail_nnaa1, id.xy * int2(1, 8) + int2({dx}, {in_g}));')
                for c_idx in range(4):
                    channel_name = ['x', 'y', 'z', 'w'][c_idx]
                    in_ch = in_g*4 + c_idx
                    ky = dy + 1
                    kx = dx + 1
                    for out_g in range(num_groups):
                        w = conv3_weights[ky, kx, in_ch, out_g*4:(out_g+1)*4]
                        lines.append(f'    f3_out_{out_g} += min16float4({fmt(w[0])},{fmt(w[1])},{fmt(w[2])},{fmt(w[3])}) * f_in.{channel_name};')
            # context up part (channels 32-63)
            for in_g in range(num_groups):
                lines.append(f'    f_in = tex2Dfetch(storageContextUp_nnaa1, id.xy * int2(1, 8) + int2({dx}, {in_g}));')
                for c_idx in range(4):
                    channel_name = ['x', 'y', 'z', 'w'][c_idx]
                    in_ch = 32 + in_g*4 + c_idx
                    ky = dy + 1
                    kx = dx + 1
                    for out_g in range(num_groups):
                        w = conv3_weights[ky, kx, in_ch, out_g*4:(out_g+1)*4]
                        lines.append(f'    f3_out_{out_g} += min16float4({fmt(w[0])},{fmt(w[1])},{fmt(w[2])},{fmt(w[3])}) * f_in.{channel_name};')
    # prelu
    alpha3 = prelu3_alpha.reshape(-1)
    for g in range(num_groups):
        for c_idx, c_name in enumerate(['x', 'y', 'z', 'w']):
            ch = g*4 + c_idx
            lines.append(f'    if(f3_out_{g}.{c_name} < 0) f3_out_{g}.{c_name} *= {fmt(alpha3[ch])};')
    # 1x1 conv (output 1 channel)
    # bias is scalar
    fb = float(final_conv_bias[0])
    lines.append(f'    min16float4 result = min16float4({fmt(fb)},{fmt(fb)},{fmt(fb)},{fmt(fb)});')
    for in_g in range(num_groups):
        lines.append(f'    f_in = f3_out_{in_g};')
        for c_idx in range(4):
            in_ch = in_g*4 + c_idx
            w = final_conv_weights[0, 0, in_ch, 0]
            lines.append(f'    result += min16float4({fmt(w)}) * f_in.{["x","y","z","w"][c_idx]};')
    # upscale to full-res (nearest neighbor)
    lines.append('    min16float val = result.x;')
    lines.append('    tex2Dstore(storageResult_nnaa1, id.xy * 2 + uint2(0,0), val);')
    lines.append('    tex2Dstore(storageResult_nnaa1, id.xy * 2 + uint2(1,0), val);')
    lines.append('    tex2Dstore(storageResult_nnaa1, id.xy * 2 + uint2(0,1), val);')
    lines.append('    tex2Dstore(storageResult_nnaa1, id.xy * 2 + uint2(1,1), val);')
    lines.append('}')
    return '\n'.join(lines)

def generate_technique():
    """Generate the technique block that dispatches all passes."""
    return """

#if (BUFFER_WIDTH % (2 * K_SIZE)) == 0
#define X_DISPATCH (BUFFER_WIDTH / (2 * K_SIZE))
#else
#define X_DISPATCH ((BUFFER_WIDTH / (2 * K_SIZE)) + 1)
#endif

#define Y_DISPATCH (BUFFER_HEIGHT / 2)
#define QX_DISPATCH (BUFFER_WIDTH / (4 * K_SIZE) + 1)
#define QY_DISPATCH (BUFFER_HEIGHT / 4)

technique Sarenya_NNAA < ui_tooltip = "Sar\\xe9nya NNAA (detail restoration)"; >
{
    pass luma
    {
        ComputeShader = GetLuma<16, 16>;
        DispatchSizeX = BUFFER_WIDTH / (2 * 16) + 1;
        DispatchSizeY = BUFFER_HEIGHT / (2 * 16) + 1;
    }

    pass detail1
    {
        ComputeShader = Layer_detail_1<K_SIZE, 1>;
        DispatchSizeX = X_DISPATCH;
        DispatchSizeY = Y_DISPATCH;
    }

    pass detail2
    {
        ComputeShader = Layer_detail_2<K_SIZE, 1>;
        DispatchSizeX = X_DISPATCH;
        DispatchSizeY = Y_DISPATCH;
    }

    pass context0
    {
        ComputeShader = Layer_context0<K_SIZE, 1>;
        DispatchSizeX = QX_DISPATCH;
        DispatchSizeY = QY_DISPATCH;
    }

    pass resblock1
    {
        ComputeShader = Layer_resblock_1<K_SIZE, 1>;
        DispatchSizeX = QX_DISPATCH;
        DispatchSizeY = QY_DISPATCH;
    }

    pass resblock2
    {
        ComputeShader = Layer_resblock_2<K_SIZE, 1>;
        DispatchSizeX = QX_DISPATCH;
        DispatchSizeY = QY_DISPATCH;
    }

    pass resblock3
    {
        ComputeShader = Layer_resblock_3<K_SIZE, 1>;
        DispatchSizeX = QX_DISPATCH;
        DispatchSizeY = QY_DISPATCH;
    }

    pass upsample_conv
    {
        ComputeShader = Layer_upsample_conv<K_SIZE, 1>;
        DispatchSizeX = X_DISPATCH;
        DispatchSizeY = Y_DISPATCH;
    }

    pass fusion
    {
        ComputeShader = Layer_fusion<K_SIZE, 1>;
        DispatchSizeX = X_DISPATCH;
        DispatchSizeY = Y_DISPATCH;
    }

    pass apply_nn
    {
        VertexShader = PostProcessVS;
        PixelShader = ApplyNN;
    }
}
"""

def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else 'nnaa.keras'
    output_path = sys.argv[2] if len(sys.argv) > 2 else 'out_nnaa.fx'

    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    model.summary()

    # Walk through layers, fold BN into Conv2D
    layers = model.layers
    ops = []
    i = 0
    while i < len(layers):
        layer = layers[i]
        if isinstance(layer, tf.keras.layers.Conv2D):
            weights = layer.get_weights()  # [kernel, bias]
            if i+1 < len(layers) and isinstance(layers[i+1], tf.keras.layers.BatchNormalization):
                bn_weights = layers[i+1].get_weights()
                new_w, new_b = fold_batch_norm(weights[0], weights[1], bn_weights)
                ops.append(('conv', new_w, new_b))
                i += 2
            else:
                ops.append(('conv', weights[0], weights[1]))
                i += 1
        elif isinstance(layer, tf.keras.layers.PReLU):
            alpha = layer.get_weights()[0]
            ops.append(('prelu', alpha))
            i += 1
        elif isinstance(layer, (tf.keras.layers.Add, tf.keras.layers.Concatenate,
                                 tf.keras.layers.UpSampling2D, tf.keras.layers.Activation)):
            i += 1
        else:
            i += 1

    # Extract operations in expected order (from the training script)
    # Architecture: detail_conv1, prelu, detail_conv2, prelu, context0_conv, prelu,
    # then 3 residual blocks (each: conv, prelu, conv, prelu), then upsample_conv, prelu,
    # then fusion_conv, prelu, final_1x1_conv (no prelu)
    idx = 0
    def next_conv():
        nonlocal idx
        op = ops[idx]
        assert op[0] == 'conv'
        idx += 1
        return op[1], op[2]  # weights, bias
    def next_prelu():
        nonlocal idx
        op = ops[idx]
        assert op[0] == 'prelu'
        idx += 1
        return op[1]

    d1_w, d1_b = next_conv()
    d1_a = next_prelu()
    d2_w, d2_b = next_conv()
    d2_a = next_prelu()

    c0_w, c0_b = next_conv()
    c0_a = next_prelu()

    res_blocks = []
    for _ in range(3):
        w1, b1 = next_conv()
        a1 = next_prelu()
        w2, b2 = next_conv()
        a2 = next_prelu()
        res_blocks.append((w1, b1, a1, w2, b2, a2))

    up_w, up_b = next_conv()
    up_a = next_prelu()

    f3_w, f3_b = next_conv()
    f3_a = next_prelu()

    final_w, final_b = next_conv()  # no prelu after

    # Generate shader using previously defined functions
    parts = []
    parts.append(generate_header())
    parts.append(generate_detail_conv_1(d1_w, d1_b, d1_a))
    parts.append('\n' + generate_detail_conv_2(d2_w, d2_b, d2_a))
    parts.append('\n' + generate_context_conv0(c0_w, c0_b, c0_a))

    # Generate residual blocks with alternating storage
    src_storage = 'storageContext0_nnaa1'
    dst_storage = 'storageContext1_nnaa1'
    for i, (w1, b1, a1, w2, b2, a2) in enumerate(res_blocks):
        parts.append('\n' + generate_residual_block(i+1, w1, b1, a1, w2, b2, a2, src_storage, dst_storage))
        src_storage, dst_storage = dst_storage, src_storage

    parts.append('\n' + generate_upsample_conv(up_w, up_b, up_a))
    parts.append('\n' + generate_fusion(f3_w, f3_b, f3_a, final_w, final_b))
    parts.append(generate_technique())

    shader_code = '\n'.join(parts)

    print(f"\nWriting shader to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(shader_code)

    print(f"Done! Generated {len(shader_code)} bytes, {shader_code.count(chr(10))} lines.")


if __name__ == '__main__':
    main()
