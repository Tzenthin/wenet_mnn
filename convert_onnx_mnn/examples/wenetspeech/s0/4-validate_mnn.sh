. ./path.sh
#exp=exp/unified_conformer  # Change it to your experiment dir
exp=exp/unified_conformer_tiny3  # Change it to your experiment dir
onnx_dir=$exp/onnx
mkdir -p $onnx_dir
#python wenet/bin/validate_onnx_mnn_expr.py \
python wenet/bin/validate_mnn.py \
      --onnx_model_path /data/kzx/mnn_deploy/wenet220704/examples/wenetspeech/s0/exp/unified_conformer_tiny3/onnx/encoder_encoder.onnx \
      --mnn_model_path /data/kzx/work/MNN/build/MNN_Models/encoder_encoder.mnn \
      --config $exp/train.yaml \
      --checkpoint $exp/final.pt \
      --chunk_size 16 \
      --output_dir $onnx_dir \
      --num_decoding_left_chunks 1
