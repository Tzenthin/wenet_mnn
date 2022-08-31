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
    parser.add_argument('--chunk_size', required=True, type=int, help='decoding chunk size')
    parser.add_argument('--checkpoint', required=True, help='checkpoint model')
    parser.add_argument('--output_dir', required=True, help='output directory')
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


def export_ctc(asr_model, args):
    print("Stage-2: export ctc")
    ctc = asr_model.ctc
    ctc.forward = ctc.log_softmax
    ctc_outpath = os.path.join(args['output_dir'], 'ctc.onnx')

    print("\tStage-2.1: prepare inputs for ctc")
    #hidden = torch.randn(
    hidden = torch.ones(
        (args['batch'], args['chunk_size'] if args['chunk_size'] > 0 else 16,
         args['output_size']))
    print('输入：', hidden)
    print("\tStage-2.2: torch.onnx.export")
    dynamic_axes = {'hidden': {1: 'T'}, 'probs': {1: 'T'}}
    torch.onnx.export(
        ctc, hidden, ctc_outpath, opset_version=13,
        export_params=True, do_constant_folding=True,
        input_names=['hidden'], output_names=['probs']) #,
        #dynamic_axes=dynamic_axes, verbose=False)
    onnx_ctc = onnx.load(ctc_outpath)
    for (k, v) in args.items():
        meta = onnx_ctc.metadata_props.add()
        meta.key, meta.value = str(k), str(v)
    onnx.checker.check_model(onnx_ctc)
    onnx.helper.printable_graph(onnx_ctc.graph)
    onnx.save(onnx_ctc, ctc_outpath)
    print_input_output_info(onnx_ctc, "onnx_ctc")
    print('\t\tExport onnx_ctc, done! see {}'.format(ctc_outpath))

    print("\tStage-2.3: check onnx_ctc and torch_ctc")
    torch_output = ctc(hidden)
    print('TORCH 输出：', torch_output.shape)
    print('TORCH 输出：', torch_output)
    ort_session = onnxruntime.InferenceSession(ctc_outpath)
    onnx_output = ort_session.run(None, {'hidden': to_numpy(hidden)})
    print('ONNX 输出：', onnx_output)
    np.testing.assert_allclose(to_numpy(torch_output), onnx_output[0],
                               rtol=1e-03, atol=1e-05)
    print("\n\t\tCheck onnx_ctc, pass!\n")
    

    # 使用creatsession的方法验证ctc
    '''
    mnn_interpreter = MNN.Interpreter("/data/kzx/work/MNN/build/MNN_Models/ctc.mnn")
    print('载入MNN模型完成！！！')
    mnn_session = mnn_interpreter.createSession()
    mnn_hidden = MNN.Tensor((1, 16,512), MNN.Halide_Type_Float, to_numpy(hidden), MNN.Tensor_DimensionType_Caffe)
    input_tensors = mnn_interpreter.getSessionInputAll(mnn_session)
    input_tensors['hidden'].copyFrom(mnn_hidden)
    mnn_interpreter.runSession(mnn_session)
    mnn_probs = mnn_interpreter.getSessionOutput(mnn_session, 'probs')
    mnn_probs = np.array(mnn_probs.getData())
    print(mnn_probs)
    assert 0==1
    '''
   
    # 使用Expr的方法验证ctc部分的推理
    '''
    mnn_hidden = to_numpy(hidden)
    vars = F.load_as_dict("/data/kzx/work/MNN/build/MNN_Models/ctc.mnn")
    mnn_in_hidden = vars["hidden"]
    print(mnn_in_hidden.shape, mnn_in_hidden.data_format)
    if mnn_in_hidden.data_format == F.NC4HW4:
        mnn_in_hidden.reorder(F.NCHW)

    mnn_in_hidden.write(mnn_hidden.tolist())
    mnn_probs = vars['probs']
    print(mnn_probs.shape, mnn_probs.data_format)
    #print(mnn_prob.get_shape())
    print('MNN 输出：', mnn_probs)
    print('numpy array:', np.array(mnn_probs))
    '''


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
    torch.onnx.export(
        decoder, inputs, decoder_outpath, opset_version=13,
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
    onnx.save(onnx_decoder, decoder_outpath)
    print_input_output_info(onnx_decoder, "onnx_decoder")
    print('\t\tExport onnx_decoder, done! see {}'.format(
        decoder_outpath))

    print("\tStage-3.3: check onnx_decoder and torch_decoder")
    torch_score, torch_r_score = decoder(
        hyps, hyps_lens, encoder_out, args['reverse_weight'])
    ort_session = onnxruntime.InferenceSession(decoder_outpath)
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
    #print(model)

    arguments = {}
    arguments['output_dir'] = output_dir
    arguments['batch'] = 1
    arguments['chunk_size'] = args.chunk_size
    arguments['output_size'] = configs['encoder_conf']['output_size']
    '''
    arguments['num_blocks'] = configs['encoder_conf']['num_blocks']
    arguments['cnn_module_kernel'] = configs['encoder_conf']['cnn_module_kernel']
    arguments['head'] = configs['encoder_conf']['attention_heads']
    arguments['left_chunks'] = args.num_decoding_left_chunks
    arguments['reverse_weight'] = args.reverse_weight
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
    '''

    export_ctc(model, arguments)
    #export_decoder(model, arguments)


if __name__ == '__main__':
    main()
