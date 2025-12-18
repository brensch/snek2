import onnx

model = onnx.load("models/snake_net.onnx")
print("Input inputs:")
for input in model.graph.input:
    print(input.name, [d.dim_value if d.dim_value > 0 else d.dim_param for d in input.type.tensor_type.shape.dim])

print("\nOutput outputs:")
for output in model.graph.output:
    print(output.name, [d.dim_value if d.dim_value > 0 else d.dim_param for d in output.type.tensor_type.shape.dim])
