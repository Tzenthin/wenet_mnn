import MNN.expr as F
vars = F.load_as_dict("/data/kzx/work/MNN/build/MNN_Models/encoder.mnn")

chunk = vars["chunk"]
att_cache = vars["att_cache"]
offset = vars["offset"]
cnn_cache = vars["cnn_cache"]
required_cache_size = vars["required_cache_size"]

print(chunk.shape, chunk.data_format)
print(att_cache.shape, att_cache.data_format)
print(offset.shape, offset.data_format)
print(cnn_cache.shape, cnn_cache.data_format)
print(required_cache_size.shape, required_cache_size.data_format)

with open('t,txt', 'w') as f:
    for k in vars.keys():
        #print(k)
        f.write(k+'\n')
    #print(k.shape, k.data_format)
#assert 0==1

# 推理阶段
def mnn_inference(image, m_path):
    






# 查看输入信息
assert 0==1

# 修改原始模型的 NC4HW4 输入为 NCHW，便于输入
if (chunk.data_format == F.NC4HW4):
    chunk.reorder(F.NCHW)
        # 写入数据
    chunk.write(numpy.ones([1, 3, 224, 224], dtype="float32").tolist())


