#include <memory>
#include <string>
#include <vector>
#include "utils/mnn_utils.h"
#include "decoder/mnn_decoder_expr.h"
#include<iostream>
#include<fstream>

namespace wenet {

MNNDecoder::MNNDecoder(std::shared_ptr<FeaturePipeline> feature_pipeline): feature_pipeline_(std::move(feature_pipeline))
  {
	  
    encoder_model_Map = MNN::Express::Variable::loadMap(encoder_model_path);
    ctc_model_Map = MNN::Express::Variable::loadMap(ctc_model_path);
    std::cout<< "完成模型加载"<<std::endl;
    searcher_.reset(new CtcPrefixBeamSearch(ctc_search_opt_)); //context_graph==nllptr

    std::ifstream fin;
    fin.open(words_table, std::ios::in);
    if (!fin.is_open()) {std::cout << "无法找到这个文件！" << std::endl;}
    std::string buff;
    int c=0;
    while(getline(fin, buff)){
      std::string word="";
      for(int i = 0; i < buff.size(); i++) { 
        if(buff[i]!=' '){
	  //std::cout<<buff[i]<<std::endl;
	  word +=(buff[i]);
	 }
	else break;
        }
      //std::cout<<word<<std::endl;
      words_map.insert(make_pair(c, word));
      c++;
    }
    fin.close();
    //std::cout<<"验证map："<<words_map[1720]<<std::endl;
    
  }


void MNNDecoder::Reset()
  {
    offset_ = 100;
    att_cache_.resize(num_blocks_ * head_ * required_cache_size_ * encoder_output_size_ / head_ * 2,  0.0);
    cnn_cache_.resize(num_blocks_ * encoder_output_size_ * (cnn_module_kernel_ - 1), 0.0);
    searcher_->Reset();
  }



void MNNDecoder::ModelForward(const std::vector<std::vector<float>>& chunk_feats, std::vector<std::vector<float>> *ctc_prob)
  {
    //encoder输入
    auto input_chunk_nchw = _ChangeInputFormat(encoder_model_Map["chunk"], MNN::Express::NCHW);
    encoder_model_Map["chunk"] = input_chunk_nchw;
    float* input_chunk_Ptr = encoder_model_Map["chunk"]->writeMap<float>();
    size_t input_chunk_Size = encoder_model_Map["chunk"]->getInfo()->size;
    if(input_chunk_Size!=num_frames_*feature_dim_){
      std::cout<<"网络的chunk输入维度与输入参数不匹配！"<<input_chunk_Size<<" "<<num_frames_*feature_dim_<<std::endl<<std::endl;
    }
    int c=0;
    for (int i=0; i<chunk_feats.size(); i++) {
	for(size_t j=0;j<feature_dim_;j++){
	  input_chunk_Ptr[c]=chunk_feats[i][j];
	  c++;
	}
    }

    int* input_offset_Ptr = encoder_model_Map["offset"]->writeMap<int>();
    input_offset_Ptr[0]=offset_;
    float* input_att_cache_Ptr = encoder_model_Map["att_cache"]->writeMap<float>(); 
    size_t input_att_cache_Size = encoder_model_Map["att_cache"]->getInfo()->size;
    if(input_att_cache_Size!=att_cache_.size()){
      std::cout<<"网络的att_cache输入维度与输入参数不匹配！"<<std::endl<<std::endl<<std::endl;
    }
    for(int i=0;i<input_att_cache_Size;i++){
      input_att_cache_Ptr[i]=att_cache_[i];
    }
    float* input_cnn_cache_Ptr = encoder_model_Map["cnn_cache"]->writeMap<float>(); 
    size_t input_cnn_cache_Size = encoder_model_Map["cnn_cache"]->getInfo()->size;
    if(input_cnn_cache_Size!=cnn_cache_.size()){
      std::cout<<"网络的cnn_cache输入维度与输入参数不匹配！"<<std::endl<<std::endl<<std::endl;
    }
    for(int i=0;i<input_cnn_cache_Size;i++){
      input_cnn_cache_Ptr[i]=cnn_cache_[i];
    }
    //std::cout<<"encoder赋值结束！"<<std::endl;
    //encoder输出
    auto output_Ptr = encoder_model_Map["output"]->readMap<float>();
    auto output_Size = encoder_model_Map["output"]->getInfo()->size;
    auto r_att_cache_Ptr = encoder_model_Map["r_att_cache"]->readMap<float>();
    auto r_att_cache_Size = encoder_model_Map["r_att_cache"]->getInfo()->size;
    if(r_att_cache_Size!=att_cache_.size()){
      std::cout<<"网络的att_cache输出维度与输出参数不匹配！"<<std::endl<<std::endl<<std::endl;
    }
    for(int i=0;i<r_att_cache_Size;i++){att_cache_[i]=r_att_cache_Ptr[i];}
    auto r_cnn_cache_Ptr = encoder_model_Map["r_cnn_cache"]->readMap<float>();
    auto r_cnn_cache_Size = encoder_model_Map["r_cnn_cache"]->getInfo()->size;
    if(r_cnn_cache_Size!=cnn_cache_.size()){
      std::cout<<"网络的cnn_cache输出维度与输出参数不匹配！"<<std::endl<<std::endl<<std::endl;
    }
    for(int i=0;i<r_cnn_cache_Size;i++){cnn_cache_[i]=r_cnn_cache_Ptr[i];}
    //std::cout<<"encoder计算结束！"<<std::endl;
    //ctc输入
    float* input_ctc_Ptr = ctc_model_Map["hidden"]->writeMap<float>();
    size_t input_ctc_Size = ctc_model_Map["hidden"]->getInfo()->size;
    if(input_ctc_Size!=16*512){std::cout<<"encoder的output输出有问题！"<<std::endl;}
    ::memcpy(input_ctc_Ptr, output_Ptr, sizeof(float)*input_ctc_Size);
    //for(int i=0;i<input_ctc_Size;i++){std::cout<<i<<" "<<input_ctc_Ptr[i]<<" ";}
    //std::cout<<"ctc赋值结束！"<<std::endl;
    //ctc输出
    //std::cout<<"CTC输入size:"<<input_ctc_Size<<std::endl;
    auto output_ctc_Ptr = ctc_model_Map["probs"]->readMap<float>();
    size_t output_ctc_Size = ctc_model_Map["probs"]->getInfo()->size;
    //std::cout<<"ctc计算结束！"<<std::endl;

    
    int num_outputs = required_cache_size_; //16 //output_probs->getDimensionType()[0];
    ctc_prob->resize(num_outputs);
    for (int i = 0; i < num_outputs; i++) {
      (*ctc_prob)[i].resize(output_dim_);
      ::memcpy((*ctc_prob)[i].data(),  output_ctc_Ptr + i * output_dim_, sizeof(float) * output_dim_);
    }

  }


void MNNDecoder::Decode()
  {
    bool end_flag=false;
    while(!end_flag){
      //这里主要实现的是，读取一段音频，对音频进行每67个frame一次送入forward，
      std::vector<std::vector<float>> chunk_feats;
      if (!feature_pipeline_->Read(num_frames_, &chunk_feats)) //说明feat结束，没有获取67个frame数据，则自动补0
        {
  	   int padding_len = num_frames_ - chunk_feats.size();
  	   std::vector<float> zero_vector(feature_dim_, 0);
  	   for(int i=0; i<padding_len; i++)
  	       {
  	          chunk_feats.push_back(zero_vector);
  	       }
       	   end_flag=true;
  	}
      std::vector<std::vector<float>> ctc_log_probs;
      ModelForward(chunk_feats, &ctc_log_probs);
      //推理阶段结束，后处理进行解码
      searcher_->Search(ctc_log_probs);
      UpdateResult(); 
      std::cout<<"partial解码："<<result_[0].sentence<<std::endl;
    }
  } // MNNDecoder::Decode() end!

void MNNDecoder::UpdateResult(){
    const auto& hypotheses = searcher_->Outputs();
    const auto& inputs = searcher_->Inputs();
    const auto& likelihood = searcher_->Likelihood();
    const auto& times = searcher_->Times();
    result_.clear();
    for (size_t i = 0; i < hypotheses.size(); i++) {
      const std::vector<int>& hypothesis = hypotheses[i];
      DecodeResult path;
      path.score = likelihood[i];
      int offset = global_frame_offset_ * feature_frame_shift_in_ms();
      for (size_t j = 0; j < hypothesis.size(); j++) {
	std::string word = words_map[hypothesis[j]];
        path.sentence += (word);	  
      }
      result_.emplace_back(path);  
    }
  }





} //namespace wenet
