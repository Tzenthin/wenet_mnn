// Copyright 2020 Mobvoi Inc. All Rights Reserved.
// Author: binbinzhang@mobvoi.com (Binbin Zhang)
//         di.wu@mobvoi.com (Di Wu)

#include <iomanip>
#include <utility>

#include "decoder/params.h"
#include "frontend/wav.h"
#include "decoder/mnn_decoder_expr.h"
#include "utils/mnn_utils.h"
#include<iostream>

int main(int argc, char *argv[]) {
  auto feature_config = wenet::InitFeaturePipelineConfigFromFlags();
  if (argc != 2){ 
	  std::cout<<"You should use only two arguments!"<< std::endl;
    return -1;
  }

  std::string wav_path = argv[1]; //第0个参数为函数名，第一个参数为解码的文件名
  std::vector<std::pair<std::string, std::string>> waves;
  if (!wav_path.empty()) {
    waves.emplace_back(make_pair("test", wav_path));
  } 

  for (auto &wav : waves) {
    wenet::WavReader wav_reader(wav.second);

    auto feature_pipeline = std::make_shared<wenet::FeaturePipeline>(*feature_config);
    feature_pipeline->AcceptWaveform(std::vector<float>(wav_reader.data(), wav_reader.data() + wav_reader.num_sample()));
    int num_frames = feature_pipeline->num_frames();
    std::cout<<"音频的frame数： "<<num_frames<<std::endl;
    std::cout<<"测试音频！ "<<wav.first<< wav.second << std::endl;
    wenet::MNNDecoder decoder(feature_pipeline); 
    std::cout<<"验证！ "<<wav.first<< wav.second << std::endl;
    decoder.Reset();    
    decoder.print_elements();
    decoder.Decode();
    std::string result = decoder.get_result();
    std::cout << "最终解码结果：" << result << std::endl;
  }
  return 0;
}
