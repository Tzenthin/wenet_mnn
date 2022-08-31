#ifndef DECODER_MNN_DECODER_EXPR_H_
#define DECODER_MNN_DECODER_EXPR_H_

#include <memory>
#include <string>
#include <vector>
#include "decoder/ctc_prefix_beam_search.h"
#include "decoder/search_interface.h"
#include "utils/mnn_utils.h"
#include "frontend/mnn_feature_pipeline.h"
//#include "MNN/Interpreter.hpp"
//#include "MNN/Tensor.hpp"
#include<iostream>
#include "MNN/expr/Expr.hpp"
#include "MNN/expr/ExprCreator.hpp"
#include "MNN/expr/Module.hpp"
#include "MNN/expr/NeuralNetWorkOp.hpp"
#include <fstream>


namespace wenet {

struct ModelOptions {
  int eos = 5537;
  int sos = 5537;

};


struct WordPiece {
  std::string word;
  int start = -1;
  int end = -1;
  WordPiece(std::string word, int start, int end): word(std::move(word)), start(start), end(end) {}
};

struct DecodeResult {
  float score = -kFloatMax;
  std::string sentence;
  std::vector<WordPiece> word_pieces;
  
  static bool CompareFunc(const DecodeResult& a, const DecodeResult& b) {
    return a.score > b.score;
  }
};


class MNNDecoder {
  public:
     MNNDecoder(std::shared_ptr<FeaturePipeline> feature_pipeline);//, const DecodeOptions& opts);
     void Decode();
     void Reset();
     void UpdateResult();
     void increae_offset() {offset_ += required_cache_size_;}
     void print_elements(){
         std::cout<<"encoder_model_path: "<<encoder_model_path<<std::endl;
         std::cout<<"ctc_model_path: "<<ctc_model_path<<std::endl;
       }
     std::string get_result() {return result_[0].sentence;}

  private:
     const char* encoder_model_path = "/data/kzx/work/MNN/build/MNN_Models/encoder_encoder_fp16.mnn"; 
     //const char* encoder_model_path = "/data/kzx/work/MNN/build/MNN_Models/encoder_encoder.mnn"; 
     const char* ctc_model_path = "/data/kzx/work/MNN/build/MNN_Models/ctc.mnn";
     const char* words_table = "/data/kzx/work/MNN/asr_mnn_proj/words.txt" ;
     const int num_blocks_ = 3;  // 蒸馏过的小模型，层数为3
     const int head_ = 8; // 多层注意力头
     const int required_cache_size_ = 16;
     const int cnn_module_kernel_ = 15;
     const int encoder_output_size_ = 512;
     const int feature_dim_ = 80;
     const int num_frames_ = 67;
     const int output_dim_ = 5538;
     std::shared_ptr<FeaturePipeline> feature_pipeline_;

     void ModelForward(const std::vector<std::vector<float>>& chunk_feats, std::vector<std::vector<float>> *ctc_prob);
     
     std::map<std::string, MNN::Express::VARP> encoder_model_Map;   
     std::map<std::string, MNN::Express::VARP> ctc_model_Map;     


     int offset_=100;
     std::vector<float> att_cache_, cnn_cache_;

     CtcPrefixBeamSearchOptions ctc_search_opt_;
     std::unique_ptr<SearchInterface> searcher_;
     std::vector<DecodeResult> result_;
     int global_frame_offset_ = 0;
     int feature_frame_shift_in_ms() const {return feature_pipeline_->config().frame_shift * 1000 / feature_pipeline_->config().sample_rate;}
     
     std::map<int, std::string> words_map;

  };


}


#endif
