# Copyright (c) 2022, Xingchen Song (sxc19@mails.tsinghua.edu.cn)
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

from __future__ import print_function

import argparse
import os
import copy
import sys

import torch
import yaml
import numpy as np
from typing import Tuple, Union
import math
from wenet.transformer.asr_model import init_asr_model
from wenet.utils.checkpoint import load_checkpoint
import numpy
import MNN
import MNN.expr as F

try:
    import onnx
    import onnxruntime
except ImportError:
    print('Please install onnx and onnxruntime!')
    sys.exit(1)


def get_args():
    parser = argparse.ArgumentParser(description='export your script model')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--checkpoint', required=True, help='checkpoint model')
    parser.add_argument('--output_dir', required=True, help='output directory')
    parser.add_argument('--chunk_size', required=True,
                        type=int, help='decoding chunk size')
    parser.add_argument('--num_decoding_left_chunks', required=True,
                        type=int, help='cache chunks')
    parser.add_argument('--reverse_weight', default=0.5,
                        type=float, help='reverse_weight in attention_rescoing')
    args = parser.parse_args()
    return args


def to_numpy(tensor):
    if tensor.requires_grad:
        return tensor.detach().cpu().numpy()
    else:
        return tensor.cpu().numpy()


def print_input_output_info(onnx_model, name, prefix="\t\t"):
    input_names = [node.name for node in onnx_model.graph.input]
    input_shapes = [[d.dim_value for d in node.type.tensor_type.shape.dim]
                    for node in onnx_model.graph.input]
    output_names = [node.name for node in onnx_model.graph.output]
    output_shapes = [[d.dim_value for d in node.type.tensor_type.shape.dim]
                     for node in onnx_model.graph.output]
    print("{}{} inputs : {}".format(prefix, name, input_names))
    print("{}{} input shapes : {}".format(prefix, name, input_shapes))
    print("{}{} outputs: {}".format(prefix, name, output_names))
    print("{}{} output shapes : {}".format(prefix, name, output_shapes))


def export_encoder(asr_model, args):
    print("Stage-1: export encoder")
    encoder = asr_model.encoder
    encoder.forward = encoder.forward_chunk_encoder
    encoder_outpath = os.path.join(args['output_dir'], 'encoder_encoder.onnx')

    print("\tStage-1.1: prepare inputs for encoder")
    chunk = torch.randn((args['batch'], 1, args['decoding_window'], args['feature_size']))
    offset = torch.ones((1, ), dtype=torch.int64)*100 #tensor([100]) #required_cache_size
    required_cache_size = 16
    att_cache = torch.zeros((args['num_blocks'], args['head'], required_cache_size, args['output_size'] // args['head'] * 2))
    cnn_cache = torch.zeros((args['num_blocks'], args['batch'], args['output_size'], args['cnn_module_kernel'] - 1))
    att_mask = torch.ones((0, 0, 0), dtype=torch.bool)
    
    inputs = (chunk, offset, att_cache, cnn_cache, att_mask)

    print("\t\tchunk.size(): {}\n".format(chunk.size()),
          "\t\toffset: {}\n".format(offset))
    print("\tStage-1.2: torch.onnx.export")
    torch.onnx.export(
        encoder, inputs, encoder_outpath, opset_version=13,
        export_params=True, do_constant_folding=True,
        input_names=['chunk', 'offset', 'att_cache', 'cnn_cache', 'att_mask'],
        output_names=['output', 'r_att_cache', 'r_cnn_cache'])

    onnx_encoder = onnx.load(encoder_outpath)
    for (k, v) in args.items():
        meta = onnx_encoder.metadata_props.add()
        meta.key, meta.value = str(k), str(v)
    onnx.checker.check_model(onnx_encoder)
    onnx.helper.printable_graph(onnx_encoder.graph)
    onnx.save(onnx_encoder, encoder_outpath)
    print_input_output_info(onnx_encoder, "onnx_encoder")
    print("\tStage-1.3: check onnx_encoder and torch_encoder")
    torch_output = []
    torch_chunk = copy.deepcopy(chunk)
    torch_offset = copy.deepcopy(offset)
    torch_att_cache = copy.deepcopy(att_cache)
    torch_cnn_cache = copy.deepcopy(cnn_cache)
    torch_att_mask = copy.deepcopy(att_mask)

    for i in range(1):
        out, r_a, r_c = encoder(torch_chunk, torch_offset, torch_att_cache, torch_cnn_cache, torch_att_mask)
        print(out.shape)
        torch_output = out #.append(out)
    print('TORCH输出：', torch_output.shape, torch_output)
    onnx_output = []
    onnx_chunk = to_numpy(chunk)
    onnx_offset = to_numpy(offset)
    onnx_att_cache = to_numpy(att_cache)
    onnx_cnn_cache = to_numpy(cnn_cache)
    onnx_att_mask = to_numpy(att_mask)
    ort_session = onnxruntime.InferenceSession(encoder_outpath)
    #input_names = [node.name for node in onnx_encoder.graph.input]
    for i in range(1):
        ort_inputs = {'chunk': onnx_chunk, 'offset':onnx_offset, 'att_cache':onnx_att_cache, 'cnn_cache':onnx_cnn_cache} 
        #for k in list(ort_inputs):
        #    if k not in input_names:
        #        ort_inputs.pop(k)
        ort_outs = ort_session.run(None, ort_inputs)
        onnx_output = ort_outs[0] #.append(ort_outs[1])
        #onnx_output.append(ort_outs[0])
        #onnx_offset += ort_outs[1].shape[0]
        onnx_r_att_cache = ort_outs[1]
        onnx_r_cnn_cache = ort_outs[2]
    #onnx_output = np.concatenate(onnx_output, axis=1)
    print('ONNX输出：', onnx_output.shape, onnx_output)
    #print('ONNX的att cache输出：', onnx_r_att_cache.shape, onnx_r_att_cache)
    print('ONNX的cnn cache输出：', onnx_r_cnn_cache.shape, onnx_r_cnn_cache)
    np.testing.assert_allclose(to_numpy(torch_output), onnx_output,
                               rtol=1e-03, atol=1e-05)
    meta = ort_session.get_modelmeta()
    print("\t\tcustom_metadata_map={}".format(meta.custom_metadata_map))
    print("\n\t\tCheck onnx_encoder, pass!\n")


    ''' 
    mnn_chunk = to_numpy(chunk)
    mnn_offset = to_numpy(offset)
    mnn_att_cache = to_numpy(att_cache)
    mnn_cnn_cache = to_numpy(cnn_cache)
    #mnn_att_mask = to_numpy(att_mask)
    vars = F.load_as_dict("/data/kzx/work/MNN/build/MNN_Models/encoder_encoder.mnn")
    #print(vars.keys())
    #assert 0==1
    #
    mnn_in_chunk = vars["chunk"]
    mnn_in_offset = vars['offset']
    mnn_in_att_cache = vars['att_cache']
    mnn_in_cnn_cache = vars['cnn_cache']
    #mnn_att_mask = vars['att_mask']

    print(mnn_in_chunk.shape, mnn_in_chunk.data_format)
    #print(mnn_in_pos_emb.shape, mnn_in_pos_emb.data_format)
    print(mnn_in_offset.shape, mnn_in_offset.data_format)
    print(mnn_in_att_cache.shape, mnn_in_att_cache.data_format)
    print(mnn_in_cnn_cache.shape, mnn_in_cnn_cache.data_format)
    #assert 0==1
    
    if mnn_in_chunk.data_format == F.NC4HW4:
        mnn_in_chunk.reorder(F.NCHW)
    mnn_in_chunk.write(mnn_chunk.tolist())
    #if mnn_in_pos_emb.data_format == F.NC4HW4:
    #    mnn_in_pos_emb.recoider(F.NCHW)
    #mnn_in_pos_emb.write(mnn_pos_emb.tolist())
    mnn_in_offset.write(mnn_offset.tolist())
    mnn_in_att_cache.write(mnn_att_cache.tolist())
    mnn_in_cnn_cache.write(mnn_cnn_cache.tolist())
    #mnn_att_mask.write(mnn_att_mask.tolist())
    mnn_output = vars['output']
    mnn_r_att_cache = vars['r_att_cache']
    mnn_r_cnn_cache = vars['r_cnn_cache']
    print('输出数据格式：', mnn_output.data_format)
    #mnn_output_data = numpy.array(copy.deepcopy(mnn_output))
    print('MNN的输出：', mnn_output)
    #print('MNN的r_att_cache输出：', mnn_r_att_cache)
    print('MNN的r_cnn_cache输出：', mnn_r_cnn_cache)
    np.testing.assert_allclose(to_numpy(torch_output), mnn_output, rtol=1e-03, atol=1e-05)
    print('MNN 测试通过！！！')
    '''


def main():
    args = get_args()
    output_dir = args.output_dir
    os.system("mkdir -p " + output_dir)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    model = init_asr_model(configs)
    load_checkpoint(model, args.checkpoint)
    model.eval()
    #print(model)

    arguments = {}
    arguments['output_dir'] = output_dir
    arguments['batch'] = 1
    arguments['chunk_size'] = args.chunk_size
    arguments['left_chunks'] = args.num_decoding_left_chunks
    arguments['reverse_weight'] = args.reverse_weight
    arguments['output_size'] = configs['encoder_conf']['output_size']
    arguments['num_blocks'] = configs['encoder_conf']['num_blocks']
    arguments['cnn_module_kernel'] = configs['encoder_conf']['cnn_module_kernel']
    arguments['head'] = configs['encoder_conf']['attention_heads']
    arguments['feature_size'] = configs['input_dim']
    arguments['vocab_size'] = configs['output_dim']
    # NOTE(xcsong): if chunk_size == -1, hardcode to 67
    arguments['decoding_window'] = (args.chunk_size - 1) * \
        model.encoder.embed.subsampling_rate + \
        model.encoder.embed.right_context + 1 if args.chunk_size > 0 else 67
    arguments['encoder'] = configs['encoder']
    arguments['decoder'] = configs['decoder']
    arguments['subsampling_rate'] = model.subsampling_rate()
    arguments['right_context'] = model.right_context()
    arguments['sos_symbol'] = model.sos_symbol()
    arguments['eos_symbol'] = model.eos_symbol()
    arguments['is_bidirectional_decoder'] = 1 \
        if model.is_bidirectional_decoder() else 0

    # NOTE(xcsong): Please note that -1/-1 means non-streaming model! It is
    #   not a [16/4 16/-1 16/0] all-in-one model and it should not be used in
    #   streaming mode (i.e., setting chunk_size=16 in `decoder_main`). If you
    #   want to use 16/-1 or any other streaming mode in `decoder_main`,
    #   please export onnx in the same config.
    if arguments['left_chunks'] > 0:
        assert arguments['chunk_size'] > 0  # -1/4 not supported

    export_encoder(model, arguments)

if __name__ == '__main__':
    main()
