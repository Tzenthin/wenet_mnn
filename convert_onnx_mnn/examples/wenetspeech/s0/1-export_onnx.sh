. ./path.sh
#exp=exp/unified_conformer  # Change it to your experiment dir
exp=exp/unified_conformer_tiny3  # Change it to your experiment dir
onnx_dir=$exp/onnx
mkdir -p $onnx_dir
python wenet/bin/export_encoder.py \
      --config $exp/train.yaml \
      --checkpoint $exp/final.pt \
      --chunk_size 16 \
      --output_dir $onnx_dir \
      --num_decoding_left_chunks 1
