. ./path.sh
#exp=exp/unified_conformer  # Change it to your experiment dir
exp=exp/unified_conformer_tiny3  # Change it to your experiment dir
onnx_dir=$exp/onnx
mkdir -p $onnx_dir
python wenet/bin/export_ctc.py \
    --config $exp/train.yaml \
    --checkpoint $exp/final.pt \
    --chunk_size 16 \
    --output_dir $onnx_dir 

/data/kzx/work/MNN/build/MNNConvert -f ONNX --modelFile exp/unified_conformer_tiny3/onnx/ctc.onnx --MNNModel exp/unified_conformer_tiny3/mnn//ctc.mnn --bizCode MNN 
#/data/kzx/work/MNN/build/MNNConvert -f ONNX --modelFile exp/unified_conformer_tiny3/onnx/ctc.onnx --MNNModel exp/unified_conformer_tiny3/mnn//ctc.mnn --bizCode MNN --fp16
