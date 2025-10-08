# VideoREPA-2B sft
python generate.py \
        --input_file ./videophy.txt \
        --output_dir ./output_dir/VideoREPA_2B \
        --model_path ../ckpt/cogvideox-2b-infer \
        --generate_type t2v \
        --upsampled

# For videophy2, change videophy.txt to videophy2.csv