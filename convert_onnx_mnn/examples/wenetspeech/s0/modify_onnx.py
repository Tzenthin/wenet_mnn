import onnx

onnx_path = 'exp/unified_conformer/onnx/encoder.onnx'

onnx_model = onnx.load(onnx_path)
print("模型加载成功")
graph = onnx_model.graph
node  = graph.node

#print(graph.output[1])
print(graph.output)
assert 0==1

for i in range(len(node)):
    if node[i].op_type == 'Concat':
        node_rise = node[i]
        if node_rise.output[0] == 'r_att_cache':
            # 7代表的是int64,其他对应类型见文末
            # 1 floart32
            graph.output[1].type.tensor_type.elem_type =1
            # 12, 8, 16, 128
            graph.output[1].type.tensor_type.shape.dim[0].dim_value = 12
            graph.output[1].type.tensor_type.shape.dim[1].dim_value = 8
            graph.output[1].type.tensor_type.shape.dim[2].dim_value = 16
            graph.output[1].type.tensor_type.shape.dim[3].dim_value = 128
print("模型修改之后：\n", graph.output[1]) #.type.tensor_type.shape.dim)
onnx.checker.check_model(onnx_model)
onnx.save(onnx_model, 'exp/unified_conformer/onnx/encoder-mod.onnx')
print("模型已保存")


