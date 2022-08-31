. ./path.sh || exit 1; 
dir=exp/unified_conformer
#dir=exp/conformer
dst_dir=torch_jit
mkdir -p $dst_dir
python wenet/bin/export_jit_trace.py \
    --config $dir/train.yaml \
    --checkpoint $dir/avg10.pt \
    --output_file $dst_dir/final_jit.pt
    #--output_quant_file $dst_dir/final_quant.zip
