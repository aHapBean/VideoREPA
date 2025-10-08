python zero_to_fp32.py ./ ../transformer --safe_serialization
# rename
mv ./transformer/model.safetensors.index.json ./transformer/diffusion_pytorch_model.safetensors.index.json

