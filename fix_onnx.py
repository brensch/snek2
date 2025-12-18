import onnx

model_path = "models/snake_net.onnx"
model = onnx.load(model_path)

# Find the input batch dimension param
# It might be a param string like "batch_size" or "s77"
input_dim = model.graph.input[0].type.tensor_type.shape.dim[0]
if input_dim.HasField("dim_param"):
    batch_dim_param = input_dim.dim_param
else:
    print("Input dim 0 is not a param! Setting it to 'batch_size'")
    batch_dim_param = "batch_size"
    input_dim.dim_param = batch_dim_param
    if input_dim.HasField("dim_value"):
        input_dim.ClearField("dim_value")

print(f"Using batch dim param: {batch_dim_param}")

# Fix outputs
for output in model.graph.output:
    print(f"Fixing output {output.name}...")
    # Set the first dimension to be the same param
    output.type.tensor_type.shape.dim[0].dim_param = batch_dim_param
    # Clear dim_value if it was set
    if output.type.tensor_type.shape.dim[0].HasField("dim_value"):
        output.type.tensor_type.shape.dim[0].ClearField("dim_value")

onnx.save(model, model_path)
print("Fixed model output dimensions.")
