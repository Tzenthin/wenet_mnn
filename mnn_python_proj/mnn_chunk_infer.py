import logging
import os
import numpy as np
import yaml
import time
import copy
from collections import deque
from collections import defaultdict
from wenet.utils.file_utils import read_symbol_table
from wenet.transformer.asr_model_streaming import init_asr_model
from wenet.utils.checkpoint import load_checkpoint
import torch
import datetime
from FeaturePipeline import Feature_Pipeline
from wenet.utils.common import (IGNORE_ID, add_sos_eos, log_add,
                                remove_duplicates_and_blank, th_accuracy,
                                reverse_pad_list)
from wenet.utils.mask import (make_pad_mask, mask_finished_preds,
                              mask_finished_scores, subsequent_mask)
from torch.nn.utils.rnn import pad_sequence
import MNN
import MNN.expr as F



class ASR_Model():

    def __init__(self, model_config): #, Feat_Pipeline):
        self.configs = model_config
        symbol_table = read_symbol_table(self.configs['dict_path'])
        self.num2sym_dict = {}
        with open(self.configs['dict_path'], 'r') as fin:
            for line in fin:
                arr = line.strip().split()
                assert len(arr) == 2
                self.num2sym_dict[int(arr[1])] = arr[0]
        self.eos = len(self.num2sym_dict) - 1
        self.sos = self.eos
        self.beam_size = self.configs['beam_size']
        self.cur_hyps = [(tuple(), (0.0, -float('inf')))]

        self.feat_pipeline = Feature_Pipeline(self.configs)


    def _chunk_ctc_prefix_beam_search(self, encoder_out): #, cur_hyps): # , encoder_mask):

        # cur_hyps: (prefix, (blank_ending_score, none_blank_ending_score))
        #cur_hyps = [(tuple(), (0.0, -float('inf')))] 
        # 2. CTC beam search step by step 
        for t in range(0, maxlen):
            logp = ctc_probs[t]  # (vocab_size,)
            # key: prefix, value (pb, pnb), default value(-inf, -inf)
            next_hyps = defaultdict(lambda: (-float('inf'), -float('inf')))
            # 2.1 First beam prune: select topk best
            top_k_logp, top_k_index = logp.topk(self.beam_size)  # (beam_size,)
            for s in top_k_index:
                s = s.item()
                ps = logp[s].item()
                for prefix, (pb, pnb) in self.cur_hyps:
                    last = prefix[-1] if len(prefix) > 0 else None
                    if s == 0:  # blank
                        n_pb, n_pnb = next_hyps[prefix]
                        n_pb = log_add([n_pb, pb + ps, pnb + ps])
                        next_hyps[prefix] = (n_pb, n_pnb)
                    elif s == last:
                        #  Update *ss -> *s;
                        n_pb, n_pnb = next_hyps[prefix]
                        n_pnb = log_add([n_pnb, pnb + ps])
                        next_hyps[prefix] = (n_pb, n_pnb)
                        # Update *s-s -> *ss, - is for blank
                        n_prefix = prefix + (s, )
                        n_pb, n_pnb = next_hyps[n_prefix]
                        n_pnb = log_add([n_pnb, pb + ps])
                        next_hyps[n_prefix] = (n_pb, n_pnb)
                    else:
                        n_prefix = prefix + (s, )
                        n_pb, n_pnb = next_hyps[n_prefix]
                        n_pnb = log_add([n_pnb, pb + ps, pnb + ps])
                        next_hyps[n_prefix] = (n_pb, n_pnb)
            # 2.2 Second beam prune
            next_hyps = sorted(next_hyps.items(), 
                               key=lambda x: log_add(list(x[1])),
                               reverse=True)
            self.cur_hyps = next_hyps[:self.beam_size]
        hyps = [(y[0], log_add([y[1][0], y[1][1]])) for y in self.cur_hyps]
        return hyps #, encoder_out

    def _num2sym(self, hyps):
        content = ''
        for w in hyps:
            if w == self.eos:
                break
            content += self.num2sym_dict[w]
        return content



def to_numpy(tensor):
    if tensor.requires_grad:
        return tensor.detach().cpu().numpy()
    else:
       return tensor.cpu().numpy()
    

if __name__ == '__main__':
    audio_path = 'DEV1_1.wav' 
    audio_path = 'DEV1.wav' 
    
    with open('conf/decode_engine_V4.yaml', 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    
    asr = ASR_Model(configs)
    
    
    with open(audio_path, 'rb') as f:
        audio_byte = f.read()
    waveform = np.frombuffer(audio_byte, dtype=np.int16)
    print(waveform)
    wav_duration = len(waveform)/16000 #self.configs['engine_sample_rate_hertz']
    waveform = torch.from_numpy(waveform).float().unsqueeze(0)
    waveform_feat, feat_length = asr.feat_pipeline._extract_feature(waveform)
    print(waveform_feat.size())
        
    #encoder_model = F.load_as_dict("/data/kzx/work/MNN/build/MNN_Models/encoder_encoder.mnn")
    encoder_model = F.load_as_dict("models/encoder_encoder_fp16.mnn")
    encoder_in_chunk = encoder_model['chunk']
    encoder_in_offset = encoder_model['offset']
    encoder_in_att_cache = encoder_model['att_cache']
    encoder_in_cnn_cache = encoder_model['cnn_cache']
    ctc_model = F.load_as_dict("models/ctc.mnn")
    ctc_in = ctc_model["hidden"]

    offset = np.ones((1, ), dtype=np.int64)*100
    att_cache = np.zeros([3,8,16,128], dtype=np.float32) #: Optional[torch.Tensor] = None
    cnn_cache = np.zeros([3,1,512,14], dtype=np.float32) #: Optional[torch.Tensor] = None

    count = 1#0
    start = 0
    while start < feat_length:
        end = start + 67
        if count<1:
            chunk_feat = np.zeros([1,1,67,80])
            count+=1
        else:
            feat = waveform_feat[:, start:end, :]
            feat_len = feat.shape[1]
            if feat_len<67:
                zero_pad = np.zeros([1, 67-feat_len, 80])
                feat = np.concatenate((feat, zero_pad), axis=1)
            chunk_feat = np.expand_dims(feat, axis=0)
            start = end #-1
            count+=1

        if encoder_in_chunk.data_format == F.NC4HW4:
            encoder_in_chunk.reorder(F.NCHW)
        encoder_in_chunk.write(chunk_feat.tolist())
        encoder_in_offset.write(offset.tolist())
        #if encoder_in_att_cache.data_format == F.NC4HW4:
        #    encoder_in_att_cache.reorder(F.NCHW)
        encoder_in_att_cache.write(att_cache.tolist())
        #if encoder_in_cnn_cache.data_format == F.NC4HW4:
        #    encoder_in_cnn_cache.reorder(F.NCHW)
        encoder_in_cnn_cache.write(cnn_cache.tolist())
        
        y = encoder_model['output'].read()
        att_cache = encoder_model['r_att_cache'].read() #.fix_as_const()#.read()
        cnn_cache = encoder_model['r_cnn_cache'].read() #.fix_as_const()#.read()
        
        att_cache = copy.deepcopy(att_cache) #必须copy一份数据
        cnn_cache = copy.deepcopy(cnn_cache)
        cnn_cache = np.expand_dims(cnn_cache, 1)

        offset += y.shape[1]
        encoder_out = y #torch.from_numpy(y)

        maxlen = encoder_out.shape[1] #.size(1)
        ctc_in.write(encoder_out.tolist())
        ctc_probs = ctc_model['probs']
        ctc_probs = ctc_probs.read()
        ctc_probs = np.squeeze(ctc_probs, axis=0)
        ctc_probs = torch.from_numpy(ctc_probs)
        #print('ctc prob：', ctc_probs)
        hyps = asr._chunk_ctc_prefix_beam_search(y)
        #print('hyps:',hyps)
        hyps = list(hyps[0][0]) #.tolist()
        result = asr._num2sym(hyps)
        result = 'partial'+'+++'+result
        print('advance decoding: ', result)


