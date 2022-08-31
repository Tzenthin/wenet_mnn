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

from wenet.transformer.asr_model import init_asr_model
from wenet.utils.checkpoint import load_checkpoint

import MNN

try:
    import onnx
    import onnxruntime
    #import onnxsim
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


def val_encoder(asr_model, args):
    #print("Stage-1: export encoder")
    encoder = asr_model.encoder
    #encoder.forward = encoder.forward_chunk
    encoder.forward = encoder.forward_chunk_modified
    encoder_outpath = os.path.join(args['output_dir'], 'encoder.onnx')

    chunk = torch.randn((args['batch'], 1, args['decoding_window'], args['feature_size']))
    #chunk = torch.randn((1, args['decoding_window'], args['feature_size']))
    
    offset = 100
    # NOTE(xcsong): The uncertainty of `next_cache_start` only appears
    #   in the first few chunks, this is caused by dynamic att_cache shape, i,e
    #   (0, 0, 0, 0) for 1st chunk and (elayers, head, ?, d_k*2) for subsequent
    #   chunks. One way to ease the ONNX export is to keep `next_cache_start`
    #   as a fixed value. To do this, for the **first** chunk, if
    #   left_chunks > 0, we feed real cache & real mask to the model, otherwise
    #   fake cache & fake mask. In this way, we get:
    #   1. 16/-1 mode: next_cache_start == 0 for all chunks
    #   2. 16/4  mode: next_cache_start == chunk_size for all chunks
    #   3. 16/0  mode: next_cache_start == chunk_size for all chunks
    #   4. -1/-1 mode: next_cache_start == 0 for all chunks
    #   NO MORE DYNAMIC CHANGES!!
    #
    # NOTE(Mddct): We retain the current design for the convenience of supporting some
    #   inference frameworks without dynamic shapes. If you're interested in all-in-one
    #   model that supports different chunks please see:
    #   https://github.com/wenet-e2e/wenet/pull/1174

    if args['left_chunks'] > 0:  # 16/4
        required_cache_size = args['chunk_size'] * args['left_chunks']
        offset = torch.ones((1, ), dtype=torch.int)*100 #required_cache_size  #offset必须大于等于16 required_cache_size 
        # Real cache
        att_cache = torch.zeros(
            (args['num_blocks'], args['head'], required_cache_size,
             args['output_size'] // args['head'] * 2))
        # Real mask
        #att_mask = torch.ones((args['batch'], 1, required_cache_size + args['chunk_size']),dtype=torch.bool)
        #att_mask[:, :, :required_cache_size] = 0
        required_cache_size = torch.ones((1,), dtype=torch.int)* 16
        # Real mask
        att_mask = torch.ones((0, 0, 0),dtype=torch.bool)  # 流式解码。每次只使用前面的16chunk作为cache，可以用不着mask

    elif args['left_chunks'] <= 0:  # 16/-1, -1/-1, 16/0
        required_cache_size = -1 if args['left_chunks'] < 0 else 0
        # Fake cache
        att_cache = torch.zeros(
            (args['num_blocks'], args['head'], 0,
             args['output_size'] // args['head'] * 2))
        # Fake mask
        att_mask = torch.ones((0, 0, 0), dtype=torch.bool)
    
    cnn_cache = torch.zeros(
        (args['num_blocks'], args['batch'],
         args['output_size'], args['cnn_module_kernel'] - 1))
    inputs = (chunk, offset, required_cache_size,
              att_cache, cnn_cache, att_mask)
    print("\t\tchunk.size(): {}\n".format(chunk.size()),
          "\t\toffset: {}\n".format(offset),
          "\t\trequired_cache: {}\n".format(required_cache_size),
          "\t\tatt_cache.size(): {}\n".format(att_cache.size()),
          "\t\tcnn_cache.size(): {}\n".format(cnn_cache.size()),
          "\t\tatt_mask.size(): {}\n".format(att_mask.size()))

    # NOTE(xcsong): We keep dynamic axes even if in 16/4 mode, this is
    #   to avoid padding the last chunk (which usually contains less
    #   frames than required). For users who want static axes, just pop
    #   out specific axis.
    # if args['chunk_size'] > 0:  # 16/4, 16/-1, 16/0
    #     dynamic_axes.pop('chunk')
    #     dynamic_axes.pop('output')
    # if args['left_chunks'] >= 0:  # 16/4, 16/0
    #     # NOTE(xsong): since we feed real cache & real mask into the
    #     #   model when left_chunks > 0, the shape of cache will never
    #     #   be changed.
    #     dynamic_axes.pop('att_cache')
    #     dynamic_axes.pop('r_att_cache')


    onnx_encoder = onnx.load(encoder_outpath)
    #print_input_output_info(onnx_encoder, "onnx_encoder")
    #for (k, v) in args.items():
    #    meta = onnx_encoder.metadata_props.add()
    #    meta.key, meta.value = str(k), str(v)
    #onnx.checker.check_model(onnx_encoder)
    #onnx.helper.printable_graph(onnx_encoder.graph)
    # NOTE(xcsong): to add those metadatas we need to reopen
    #   the file and resave it.
    
    #onnx.save(onnx_encoder, encoder_outpath)
    torch_output = []
    torch_chunk = copy.deepcopy(chunk)
    torch_offset = copy.deepcopy(offset)
    torch_required_cache_size = copy.deepcopy(required_cache_size)
    torch_att_cache = copy.deepcopy(att_cache)
    torch_cnn_cache = copy.deepcopy(cnn_cache)
    torch_att_mask = copy.deepcopy(att_mask)
    for i in range(10):
        print("\t\ttorch chunk-{}: {}, offset: {}, att_cache: {},"
              " cnn_cache: {}, att_mask: {}".format(
                  i, list(torch_chunk.size()), torch_offset,
                  list(torch_att_cache.size()),
                  list(torch_cnn_cache.size()), list(torch_att_mask.size())))
        # NOTE(xsong): att_mask of the first few batches need changes if
        #   we use 16/4 mode.
        if args['left_chunks'] > 0:  # 16/4
            torch_att_mask[:, :, -(args['chunk_size'] * (i + 1)):] = 1
        out, torch_att_cache, torch_cnn_cache = encoder(
            torch_chunk, torch_offset, torch_required_cache_size,
            torch_att_cache, torch_cnn_cache, torch_att_mask)
        torch_output.append(out)
        torch_offset += out.size(1)
        #print('torch att_cache shape: ', torch_att_cache.size())
    torch_output = torch.cat(torch_output, dim=1)
    print(torch_output)


    onnx_output = []
    onnx_chunk = to_numpy(chunk)
    onnx_offset = to_numpy(offset) #np.array((offset)).astype(np.int64)
    onnx_required_cache_size = to_numpy(required_cache_size) #np.array((required_cache_size)).astype(np.int64)
    onnx_att_cache = to_numpy(att_cache)
    onnx_cnn_cache = to_numpy(cnn_cache)
    onnx_att_mask = to_numpy(att_mask)
    ort_session = onnxruntime.InferenceSession(encoder_outpath, providers=['CPUExecutionProvider'])
    input_names = [node.name for node in onnx_encoder.graph.input]
    print('onnx输入节点名称：',input_names)
    for i in range(10):
        print("\t\tonnx  chunk-{}: {}, offset: {}, att_cache: {},"
              " cnn_cache: {}, att_mask: {}".format(
                  i, onnx_chunk.shape, onnx_offset, onnx_att_cache.shape,
                  onnx_cnn_cache.shape, onnx_att_mask.shape))
        # NOTE(xsong): att_mask of the first few batches need changes if
        #   we use 16/4 mode.
        if args['left_chunks'] > 0:  # 16/4
            onnx_att_mask[:, :, -(args['chunk_size'] * (i + 1)):] = 1
        ort_inputs = {
            'chunk': onnx_chunk, 'offset': onnx_offset,
            'required_cache_size': onnx_required_cache_size,
            'att_cache': onnx_att_cache, 'cnn_cache': onnx_cnn_cache,
            'att_mask': onnx_att_mask
        }
        # NOTE(xcsong): If we use 16/-1, -1/-1 or 16/0 mode, `next_cache_start`
        #   will be hardcoded to 0 or chunk_size by ONNX, thus
        #   required_cache_size and att_mask are no more needed and they will
        #   be removed by ONNX automatically.
        for k in list(ort_inputs):
            if k not in input_names:
                ort_inputs.pop(k)
        ort_outs = ort_session.run(None, ort_inputs)
        onnx_att_cache, onnx_cnn_cache = ort_outs[1], ort_outs[2]
        #print('onnx att cache shape :', onnx_att_cache.shape)
        onnx_output.append(ort_outs[0])
        onnx_offset += ort_outs[0].shape[1]
        #print(ort_outs[0])
    onnx_output = np.concatenate(onnx_output, axis=1)
    np.testing.assert_allclose(to_numpy(torch_output), onnx_output,
                               rtol=1e-03, atol=1e-05)
    meta = ort_session.get_modelmeta()
    print("\t\tcustom_metadata_map={}".format(meta.custom_metadata_map))
    print("\n\t\tCheck onnx_encoder, pass!\n")


    mnn_output = []
    mnn_interpreter = MNN.Interpreter("/data/kzx/work/MNN/build/MNN_Models/encoder.mnn")
    print('载入MNN模型完成！！！')
    mnn_session = mnn_interpreter.createSession()
    #mnn_session = mnn_interpreter.resizeSession(mnn_session)

    assert 0==1
    input_tensors = mnn_interpreter.getSessionInputAll(mnn_session)
    #input_tensors['chunk'].copyFrom(mnn_chunk)
    #input_tensors['offset'].copyFrom(mnn_offset)
    #input_tensors['required_cache_size'].copyFrom(mnn_required_cache_size)
    #input_tensors['att_cache'].copyFrom(mnn_att_cache)
    #input_tensors['cnn_cache'].copyFrom(mnn_cnn_cache)
    #input_tensors['att_mask'].copyFrom(mnn_att_mask)
    #input_tensors.printTensorData() 
    #input_names = [node.name for node in onnx_encoder.graph.input]
    #input_tensor = mnn_interpreter.getSessionInput(mnn_session)
    #print('MNN 的输入shape：', input_tensor.getShape())
    #print('MNN 的输入shape：', input_tensor)
    for k in input_tensors.keys():
        print('输入：',k,'shape：', input_tensors[k].getShape())
    #offset_tmp = torch.ones(1,)*100
    #required_cache_size_tmp = torch.ones(1, )*16

    mnn_chunk = MNN.Tensor((1, 67,80), MNN.Halide_Type_Float, to_numpy(chunk), MNN.Tensor_DimensionType_Caffe)
    mnn_att_cache = MNN.Tensor((12,8,16,128), MNN.Halide_Type_Float, to_numpy(att_cache), MNN.Tensor_DimensionType_Caffe)
    mnn_cnn_cache = MNN.Tensor((12,1,512,14), MNN.Halide_Type_Float, to_numpy(cnn_cache), MNN.Tensor_DimensionType_Caffe)
    mnn_offset = MNN.Tensor((1, ), MNN.Halide_Type_Int, np.array([100]).astype(np.int32), MNN.Tensor_DimensionType_Caffe)
    #mnn_offset = MNN.Tensor((1, ), MNN.Halide_Type_Float, np.array([100]).astype(np.float32), MNN.Tensor_DimensionType_Caffe)
    assert 0==1
    mnn_required_cache_size = MNN.Tensor((1,), MNN.Halide_Type_Int, np.array([16]).astype(np.float32), MNN.Tensor_DimensionType_Caffe)
    #assert 0==1
    for i in range(1):
        input_tensors['chunk'].copyFrom(mnn_chunk)
        input_tensors['offset'].copyFrom(mnn_offset)
        input_tensors['required_cache_size'].copyFrom(mnn_required_cache_size)
        input_tensors['att_cache'].copyFrom(mnn_att_cache)
        input_tensors['cnn_cache'].copyFrom(mnn_cnn_cache)

        mnn_interpreter.runSession(mnn_session)
        output_tensors = mnn_interpreter.getSessionOutput(mnn_session, 'r_att_cache')
        output_att_cache = np.array(output_tensors.getData())
        #output_tensors = mnn_interpreter.getSessionOutputAll(mnn_session)
        print(output_att_cache)
        assert 0==1
        for k in output_tensors.keys():
            #output_tensors[k].printTensorData()
            output_data = np.array(output_tensors[k].getData())
            print(output_data)



def export_ctc(asr_model, args):
    print("Stage-2: export ctc")
    ctc = asr_model.ctc
    ctc.forward = ctc.log_softmax
    ctc_outpath = os.path.join(args['output_dir'], 'ctc.onnx')

    print("\tStage-2.1: prepare inputs for ctc")
    hidden = torch.randn(
        (args['batch'], args['chunk_size'] if args['chunk_size'] > 0 else 16,
         args['output_size']))

    print("\tStage-2.2: torch.onnx.export")
    dynamic_axes = {'hidden': {1: 'T'}, 'probs': {1: 'T'}}
    torch.onnx.export(
        ctc, hidden, ctc_outpath, opset_version=11,
        export_params=True, do_constant_folding=True,
        input_names=['hidden'], output_names=['probs'])
        #dynamic_axes=dynamic_axes, verbose=False)
    onnx_ctc = onnx.load(ctc_outpath)
    for (k, v) in args.items():
        meta = onnx_ctc.metadata_props.add()
        meta.key, meta.value = str(k), str(v)
    onnx.checker.check_model(onnx_ctc)
    onnx.helper.printable_graph(onnx_ctc.graph)
    
    
    ''' 
    #input_shape_fixed={'chunk': chunk.size(), 'att_cache':att_cache.size(), 'att_mask':att_mask.size(), 'cnn_cache':cnn_cache.size()}
    input_shape_fixed={'hidden':[1, 16,512]}
    #onnx_ctc_sim, check_ok = onnxsim.simplify(onnx_ctc, input_shapes=input_shape_fixed )#,dynamic_input_shape=True)
    #onnx.save(onnx_ctc_sim, ctc_outpath)
    #print('简化结束!!!\n')
    '''
    
    onnx.save(onnx_ctc, ctc_outpath)
    print_input_output_info(onnx_ctc, "onnx_ctc")
    print('\t\tExport onnx_ctc, done! see {}'.format(ctc_outpath))

    print("\tStage-2.3: check onnx_ctc and torch_ctc")
    torch_output = ctc(hidden)
    ort_session = onnxruntime.InferenceSession(ctc_outpath, providers=['CPUExecutionProvider'])
    onnx_output = ort_session.run(None, {'hidden': to_numpy(hidden)})

    np.testing.assert_allclose(to_numpy(torch_output), onnx_output[0],
                               rtol=1e-03, atol=1e-05)
    print("\t\tCheck onnx_ctc, pass!")


def export_decoder(asr_model, args):
    print("Stage-3: export decoder")
    decoder = asr_model
    # NOTE(lzhin): parameters of encoder will be automatically removed
    #   since they are not used during rescoring.
    decoder.forward = decoder.forward_attention_decoder
    decoder_outpath = os.path.join(args['output_dir'], 'decoder.onnx')

    print("\tStage-3.1: prepare inputs for decoder")
    # hardcode time->200 nbest->10 len->20, they are dynamic axes.
    encoder_out = torch.randn((1, 200, args['output_size']))
    hyps = torch.randint(low=0, high=args['vocab_size'],
                         size=[10, 20])
    hyps[:, 0] = args['vocab_size'] - 1  # <sos>
    hyps_lens = torch.randint(low=15, high=21, size=[10])

    print("\tStage-3.2: torch.onnx.export")
    dynamic_axes = {
        'hyps': {0: 'NBEST', 1: 'L'}, 'hyps_lens': {0: 'NBEST'},
        'encoder_out': {1: 'T'},
        'score': {0: 'NBEST', 1: 'L'}, 'r_score': {0: 'NBEST', 1: 'L'}
    }
    inputs = (hyps, hyps_lens, encoder_out, args['reverse_weight'])
    
    print("\t\thyps.size(): {}\n".format(hyps.size()),
          "\t\thyps_lens.size(): {}\n".format(hyps_lens.size()),
          "\t\tencoder_out.size(): {}\n".format(encoder_out.size()),
          "\t\treverse_weight): {}\n".format(args['reverse_weight']))
    
    torch.onnx.export(
        decoder, inputs, decoder_outpath, opset_version=11,
        export_params=True, do_constant_folding=True,
        input_names=['hyps', 'hyps_lens', 'encoder_out', 'reverse_weight'],
        output_names=['score', 'r_score'])#,
        #dynamic_axes=dynamic_axes, verbose=False)
    onnx_decoder = onnx.load(decoder_outpath)
    for (k, v) in args.items():
        meta = onnx_decoder.metadata_props.add()
        meta.key, meta.value = str(k), str(v)
    onnx.checker.check_model(onnx_decoder)
    onnx.helper.printable_graph(onnx_decoder.graph)
    
    
    '''
    input_shape_fixed={'hyps':[10, 20], 'hyps_lens':[10], 'encoder_out':[1,200, 512]}
    onnx_decoder_sim, check_ok = onnxsim.simplify(onnx_decoder, input_shapes=input_shape_fixed )#,dynamic_input_shape=True)
    onnx.save(onnx_decoder_sim, decoder_outpath)
    print('简化结束!!!\n')
    '''
    
    onnx.save(onnx_decoder, decoder_outpath)
    print_input_output_info(onnx_decoder, "onnx_decoder")
    print('\t\tExport onnx_decoder, done! see {}'.format(
        decoder_outpath))

    print("\tStage-3.3: check onnx_decoder and torch_decoder")
    torch_score, torch_r_score = decoder(
        hyps, hyps_lens, encoder_out, args['reverse_weight'])
    ort_session = onnxruntime.InferenceSession(decoder_outpath, providers=['CPUExecutionProvider'])
    input_names = [node.name for node in onnx_decoder.graph.input]
    ort_inputs = {
        'hyps': to_numpy(hyps),
        'hyps_lens': to_numpy(hyps_lens),
        'encoder_out': to_numpy(encoder_out),
        'reverse_weight': np.array((args['reverse_weight'])),
    }
    for k in list(ort_inputs):
        if k not in input_names:
            ort_inputs.pop(k)
    onnx_output = ort_session.run(None, ort_inputs)

    np.testing.assert_allclose(to_numpy(torch_score), onnx_output[0],
                               rtol=1e-03, atol=1e-05)
    if args['is_bidirectional_decoder'] and args['reverse_weight'] > 0.0:
        np.testing.assert_allclose(to_numpy(torch_r_score), onnx_output[1],
                                   rtol=1e-03, atol=1e-05)
    print("\t\tCheck onnx_decoder, pass!")


def main():
    torch.manual_seed(777)
    args = get_args()
    output_dir = args.output_dir
    os.system("mkdir -p " + output_dir)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    model = init_asr_model(configs)
    load_checkpoint(model, args.checkpoint)
    model.eval()
    print(model)
    #assert 0==1

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

    val_encoder(model, arguments)
    assert 0==1
    export_ctc(model, arguments)
    export_decoder(model, arguments)


if __name__ == '__main__':
    main()
