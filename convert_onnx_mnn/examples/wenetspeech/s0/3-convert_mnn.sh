#./data/kzx/work/MNN/build/MNNConvert -f ONNX --modelFile /data/kzx/work/wenet220704/examples/wenetspeech/s0/exp/unified_conformer_tiny3/onnx/encoder_encoder.onnx --MNNModel MNN_Models/encoder_encoder.mnn --bizCode MNN  
/data/kzx/work/MNN/build/MNNConvert -f ONNX --modelFile exp/unified_conformer_tiny3/onnx/encoder_encoder.onnx --MNNModel exp/unified_conformer_tiny3/mnn/encoder_encoder.mnn --bizCode MNN  

#对简化后的onnx转换
#./MNNConvert -f ONNX --modelFile /data/kzx/work/wenet220704/examples/wenetspeech/s0/exp/unified_conformer_tiny3/onnx/encoder_encoder_sim.onnx --MNNModel MNN_Models/encoder_encoder_sim.mnn --bizCode MNN  #onnxsim简化后的模型

#转换为fp16的MNN模型
#./MNNConvert -f ONNX --modelFile /data/kzx/work/wenet220704/examples/wenetspeech/s0/exp/unified_conformer_tiny3/onnx/encoder_encoder.onnx --MNNModel MNN_Models/encoder_encoder_fp16.mnn --bizCode MNN --fp16
