# VideoREPA-5B lora
python generate.py \
        --input_file ./videophy.txt \
        --output_dir ./output_dir/VideoREPA_5B \
        --model_path ../ckpt/cogvideox-5b \
        --lora_path ./ \
        --generate_type t2v \
        --upsampled

# For videophy2, change videophy.txt to videophy2.csv