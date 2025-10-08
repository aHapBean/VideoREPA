"""
This script demonstrates how to generate videos for multiple captions using the CogVideoX model.
Captions are read from a text file, and each generated video is saved to a specified directory.
"""

import argparse
import logging
import os
from typing import Literal, Optional
from tqdm import tqdm
import torch
import json 
import pandas as pd
from diffusers import (
    CogVideoXDPMScheduler,
    CogVideoXDDIMScheduler,
    CogVideoXImageToVideoPipeline,
    CogVideoXPipeline,
    CogVideoXVideoToVideoPipeline,
)
from diffusers.utils import export_to_video, load_image, load_video

logging.basicConfig(level=logging.INFO)

# Recommended resolution for each model (width, height)
RESOLUTION_MAP = {
    "cogvideox1.5-5b-i2v": (768, 1360),
    "cogvideox1.5-5b": (768, 1360),
    "cogvideox-5b-i2v": (480, 720),
    "cogvideox-5b": (480, 720),
    "cogvideox-2b": (480, 720),
}


def generate_video(
    pipe,  
    prompt: str,
    output_path: str,
    model_path: str,
    num_frames: int = 49,
    width: Optional[int] = None,
    height: Optional[int] = None,
    image=None,
    video=None,
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    num_videos_per_prompt: int = 1,
    generate_type: str = "t2v",
    seed: int = 42,
    fps: int = 8,
):
    model_name = model_path.split("/")[-1].lower()
    desired_resolution = RESOLUTION_MAP.get(model_name, (480, 720))
    if width is None or height is None:
        height, width = desired_resolution
    elif (height, width) != desired_resolution and generate_type != "i2v":
        height, width = desired_resolution

    # generate
    torch.manual_seed(seed)
    if generate_type == "i2v":
        video_generate = pipe(
            height=height,
            width=width,
            prompt=prompt,
            image=image,
            num_videos_per_prompt=num_videos_per_prompt,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            use_dynamic_cfg=True,
            guidance_scale=guidance_scale,
            generator=torch.Generator().manual_seed(seed),
        ).frames[0]
    elif generate_type == "t2v":
        video_generate = pipe(
            height=height,
            width=width,
            prompt=prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            use_dynamic_cfg=True,
            guidance_scale=guidance_scale,
            generator=torch.Generator().manual_seed(seed),
        ).frames[0]
    else:
        video_generate = pipe(
            height=height,
            width=width,
            prompt=prompt,
            video=video,
            num_videos_per_prompt=num_videos_per_prompt,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            use_dynamic_cfg=True,
            guidance_scale=guidance_scale,
            generator=torch.Generator().manual_seed(seed),
        ).frames[0]

    export_to_video(video_generate, output_path, fps=fps)
    logging.info(f"Saved video to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate videos from text prompts")
    parser.add_argument("--prompt", type=str, default=None, help="Single text prompt for video generation")
    parser.add_argument("--input_file", type=str, default=None, help="Text file containing captions (one per line)")
    parser.add_argument("--output_dir", type=str, default="./CogVideo2B_videos", help="Directory to save videos")
    parser.add_argument("--model_path", type=str, default="THUDM/CogVideoX1.5-5B", help="Pretrained model path")
    parser.add_argument("--generate_type", type=str, default="t2v", choices=["t2v", "i2v", "v2v"])
    parser.add_argument("--image_or_video_path", type=str, default="", help="Input image/video path (for i2v/v2v)")
    parser.add_argument("--lora_path", type=str, default=None, help="The path of the LoRA weights to be used")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--upsampled", action="store_true", default=False)
    # NOTE: please set these two parameters appropriately, aligning with training if lora used.
    parser.add_argument("--lora_alpha", default=64, type=int, help="lora alpha config during training")  
    parser.add_argument("--lora_rank", default=128, type=int, help="The rank of the LoRA weights")       

    args = parser.parse_args()
    assert args.prompt is None or args.input_file is None


    if args.input_file == "videophy2.csv":
        # VideoPhy2
        args.output_dir = args.output_dir + '/' + args.input_file.strip('./').removesuffix('.csv')
        os.makedirs(args.output_dir, exist_ok=True)
        df = pd.read_csv(args.input_file)
        captions = df['caption'].tolist()
        long_captions = df['upsampled_caption'].tolist()       
    elif args.input_file == "videophy.txt":
        # VideoPhy
        args.output_dir = args.output_dir + '/' + args.input_file.strip('./').removesuffix('.txt')
        os.makedirs(args.output_dir, exist_ok=True)

        with open(args.input_file, "r", encoding="utf-8") as f:
            captions = [line.strip() for line in f if line.strip()]
        with open("videophy_detailed.txt", "r", encoding="utf-8") as f:
            long_captions = [line.strip() for line in f if line.strip()]

    dtype = torch.bfloat16
    image, video = None, None

    if args.generate_type == "i2v":
        pipe = CogVideoXImageToVideoPipeline.from_pretrained(args.model_path, torch_dtype=dtype)
        image = load_image(args.image_or_video_path) if args.image_or_video_path else None
    elif args.generate_type == "t2v":
        # For original baseline inference
        # pipe = CogVideoXPipeline.from_pretrained(args.model_path, torch_dtype=dtype)
        
        # For VideoREPA inference
        from models.cogvideox_align import CogVideoXPipelineAlign, CogVideoXTransformer3DModelAlign
        pipe = CogVideoXPipelineAlign.from_pretrained(args.model_path, torch_dtype=dtype)
        
        # Specific process for VideoREPA-5B (LoRA)
        if not isinstance(pipe.transformer, CogVideoXTransformer3DModelAlign):
            del pipe.transformer
            pipe.transformer = CogVideoXTransformer3DModelAlign.from_pretrained(args.model_path, subfolder="transformer", align_layer=18, align_dims=[768], projector_dim=2048).to(torch.bfloat16)
    else:
        pipe = CogVideoXVideoToVideoPipeline.from_pretrained(args.model_path, torch_dtype=dtype)
        video = load_video(args.image_or_video_path) if args.image_or_video_path else None

    # load lora
    if args.lora_path:
        pipe.load_lora_weights(args.lora_path, weight_name="pytorch_lora_weights.safetensors", adapter_name="test_1")
        # pipe.fuse_lora(components=["transformer"], lora_scale=1 / lora_rank)
        pipe.fuse_lora(components=["transformer"], lora_scale = args.lora_alpha / args.lora_rank)
        assert 'alpha' not in args.output_dir
        # pipe.load_lora_weights(args.lora_path,  weight_name="pytorch_lora_weights.safetensors", adapter_name="cogvideox-lora")
        # pipe.set_adapters(["cogvideox-lora"], [lora_scaling])
        pipe.transformer.to(torch.bfloat16)
        print("lora loaded!!")
    
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

    pipe.to("cuda")  
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    if args.prompt is not None:
        output_path = f"./{'_'.join(args.prompt[:min(len(args.prompt), 20)].split(' '))}.mp4"
        generate_video(
            pipe=pipe,
            prompt=args.prompt,
            output_path=output_path,
            model_path=args.model_path,
            generate_type=args.generate_type,
            image=image,
            video=video,
            seed=args.seed,
        )
        exit(0)

    if args.input_file == "videophy2.csv":
        # VideoPhy2
        assert args.upsampled   # by default
        print(f'parameter upsampled is useless when videophy2')
        for idx, (caption, long_caption) in tqdm(enumerate(zip(captions, long_captions)), total=len(captions), desc="Generating videos"):
            output_path = os.path.join(args.output_dir, f"{'_'.join(caption.rstrip('.').split(' '))}.mp4")
            if os.path.exists(output_path) or os.path.exists(output_path.replace('.mp4', '') + '.lock'):
                print(f'{output_path} already generated or in process!')
                continue

            os.makedirs(output_path.replace('.mp4', '') + '.lock')
            generate_video(
                pipe=pipe,
                prompt=long_caption,
                output_path=output_path,
                model_path=args.model_path,
                generate_type=args.generate_type,
                image=image,
                video=video,
                seed=args.seed,
            )
            if os.path.exists(output_path.replace('.mp4', '') + '.lock'):
                os.rmdir(output_path.replace('.mp4', '') + '.lock')
            print(f'Finished generating {output_path}')        
    else:
        # VideoPhy
        assert args.upsampled   # by default
        assert len(captions) == len(long_captions)
        for idx, (caption, long_caption) in tqdm(enumerate(zip(captions, long_captions)), total=len(captions), desc="Generating videos"):
            output_path = os.path.join(args.output_dir, f"{'_'.join(caption.rstrip('.').split(' '))}.mp4")
            if os.path.exists(output_path):
                print(f'{output_path} already generated!')
                continue
            if os.path.exists(output_path.replace('.mp4', '') + '.lock'):
                print(f'{output_path} already in process, skip to another video')
                continue
                
            os.makedirs(output_path.replace('.mp4', '') + '.lock')
            if args.upsampled:
                generate_video(
                    pipe=pipe,
                    prompt=long_caption,
                    output_path=output_path,
                    model_path=args.model_path,
                    generate_type=args.generate_type,
                    image=image,
                    video=video,
                    seed=args.seed,
                )
            else:
                generate_video(
                    pipe=pipe,
                    prompt=caption,
                    output_path=output_path,
                    model_path=args.model_path,
                    generate_type=args.generate_type,
                    image=image,
                    video=video,
                    seed=args.seed,
                )                    
            if os.path.exists(output_path.replace('.mp4', '') + '.lock'):
                os.rmdir(output_path.replace('.mp4', '') + '.lock')
            print(f'Finished generating {output_path}')
