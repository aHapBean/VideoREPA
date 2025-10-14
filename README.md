# VideoREPA (NeurIPS 2025)
### [Project Page](https://videorepa.github.io) | [Paper](https://arxiv.org/abs/2505.23656)

> VideoREPA: Learning Physics for Video Generation through Relational Alignment with Foundation Models\
> Xiangdong Zhang, Jiaqi Liao, Shaofeng Zhang, Fanqing Meng, Xiangpeng Wan, Junchi Yan, Yu Cheng \
> NeurIPS 2025

✨ **A step towards more reliable world modeling by enhancing physics plausibility in video generation.**

|VideoPhy|SA|PC|
-|-|-
CogVideoX-5B | 70.0 | 32.3 |
+REPA Loss+DINOv2 | 62.5 ⚠️ | 33.7 | 
+REPA Loss+VideoMAEv2 | 59.3 ⚠️ | 35.5 |
+TRD Loss+VideoMAEv2 (ours) | **72.1** | **40.1**


### ✅ Project Status

🎉 **Accepted to NeurIPS 2025!**  

- [x] Release introduction & visual results  
- [x] Release training & inference code  
- [x] Upload checkpoints and provide reproducing tips.
- [ ] Release evaluation code. *(coming soon!)*  

If you find VideoREPA useful, please consider giving us a star ⭐ to stay updated.

### Introduction

<div align="center">
  <div style="display: flex; justify-content: space-between;">
    <img src="https://github.com/user-attachments/assets/7e65716b-27cd-45e1-b4df-1f4c1c7c3d33" alt="test" width="35%" />
    <img src="https://github.com/user-attachments/assets/1952c95f-5453-42d9-84ec-80f49565a961" alt="test" width="35%" />
  </div>
</div>

<p align="center">
  Figure 1. Evaluation of physics understanding on the Physion benchmark. The chance performance if 50%.
</p>

🔍 
**Physics Understanding Gap:** We identify an essential gap in physics understanding between self-supervised VFMs and T2V models, proposing **the first method to bridge video understanding models and T2V models**. VideoREPA demonstrates that *“understanding helps generation.”* in video generation field.

### Overview

<div align="center">
  <div style="display: flex; justify-content: space-between;">
    <img src="https://github.com/user-attachments/assets/4a55f50c-cc02-4467-8b84-4f83ed37869e" alt="test" width="70%" />
  </div>
</div>

<p align="center">
  Figure 2. Overview of VideoREPA.
</p>

VideoREPA enhances **physics plausibility** in T2V models through **Token Relation Distillation (TRD)** — a loss that aligns **pairwise token relations** between self-supervised video encoders and diffusion transformer features.

Each token learns relations about both:
- **Spatial relations** within a frame  
- **Temporal relations** across frames  

🌟 **Novelty:** VideoREPA is the **first successful adaptation of REPA into video generation** — overcoming key challenges in finetuning large pretrained video diffusion transformers and maintaining temporal consistency.



### Qualitative Results

<table align="center" style="width: 100%;">
  <tr>
    <th align="center" style="width: 25%;">CogVideoX</th>
    <th align="center" style="width: 25%;">CogVideoX+REPA loss</th>
    <th align="center" style="width: 25%;">VideoREPA</th>
    <th align="center" style="width: 25%;">Prompt</th>
  </tr>
  <tr>
    <td align="center" style="width: 25%;">
      <video width="25%" controls src="https://github.com/user-attachments/assets/b0f6b65d-3b0b-4665-88a9-8fc81a23c613" controls autoplay loop muted></video>
    </td>
    <td align="center" style="width: 25%;">
      <video width="25%" controls src="https://github.com/user-attachments/assets/00276b57-e3ea-4f30-b0a7-6522f4dedd31" controls autoplay loop muted>Your browser does not support the video tag.</video>
    </td>
    <td align="center" style="width: 25%;">
      <video width="25%" controls src="https://github.com/user-attachments/assets/92b51720-f4d8-4867-8c1c-3fb88e2f5e67" controls autoplay loop muted>Your browser does not support the video tag.</video>
    </td>    
    <td align="center" style="width: 25%;">
      Leather glove catching a hard baseball.
    </td>
  </tr>

  <tr>
    <td align="center" style="width: 25%;">
      <video width="25%" controls src="https://github.com/user-attachments/assets/a199b5ab-0829-41de-ab72-2e17ac66f069" controls autoplay loop muted></video>
    </td>
    <td align="center" style="width: 25%;">
      <video width="25%" controls src="https://github.com/user-attachments/assets/e5d61296-0b9b-4567-aa27-b986234ce870" controls autoplay loop muted>Your browser does not support the video tag.</video>
    </td>
    <td align="center" style="width: 25%;">
      <video width="25%" controls src="https://github.com/user-attachments/assets/5acfd53c-83f7-4e18-bb37-b5eec8dcf226" controls autoplay loop muted>Your browser does not support the video tag.</video>
    </td>    
    <td align="center" style="width: 25%;">
      Maple syrup drizzling from a bottle onto pancakes.
    </td>
  </tr>

  <tr>
    <td align="center" style="width: 25%;">
      <video width="25%" controls src="https://github.com/user-attachments/assets/d93283f9-9dff-41b0-8837-d93d06d06356" controls autoplay loop muted></video>
    </td>
    <td align="center" style="width: 25%;">
      <video width="25%" controls src="https://github.com/user-attachments/assets/2be513e6-8a7f-4199-bb1e-c411fcda14ac" controls autoplay loop muted>Your browser does not support the video tag.</video>
    </td>
    <td align="center" style="width: 25%;">
      <video width="25%" controls src="https://github.com/user-attachments/assets/25dca90d-d91b-4ffe-8fc9-3c340c816d95" controls autoplay loop muted>Your browser does not support the video tag.</video>
    </td>    
    <td align="center" style="width: 25%;">
      Glass shatters on the floor.
    </td>
  </tr>

  <tr>
    <td align="center" style="width: 25%;">
      <video width="25%" controls src="https://github.com/user-attachments/assets/3095d887-4cfb-4726-8152-56d6aa72de40" controls autoplay loop muted></video>
    </td>
    <td align="center" style="width: 25%;">
      <video width="25%" controls src="https://github.com/user-attachments/assets/2b96fcde-f371-400f-9459-ca223d237c73" controls autoplay loop muted>Your browser does not support the video tag.</video>
    </td>
    <td align="center" style="width: 25%;">
      <video width="25%" controls src="https://github.com/user-attachments/assets/89299f65-fb1e-4013-969f-bc1c7e715523" controls autoplay loop muted>Your browser does not support the video tag.</video>
    </td>    
    <td align="center" style="width: 25%;">
      A child runs and catches a brightly colored frisbee...
    </td>
  </tr>

</table>

## ⚙️ Quick start

### Environment setup

```bash
git clone https://github.com/aHapBean/VideoREPA.git

conda create --name videorepa python=3.10
conda activate videorepa

cd VideoREPA
pip install -r requirements.txt

# Install diffusers locally (recommended)
cd ./finetune/diffusers
pip install -e .
```

### Dataset download
Download the [OpenVid](https://github.com/NJU-PCALab/OpenVid-1M) dataset used in VideoREPA. We use parts 30–49 and select subsets containing 32K and 64K videos, respectively. The corresponding CSV files are located in `./finetune/openvid/`.

```bash
pip install -U huggingface_hub

# Download parts 30–49
huggingface-cli download --repo-type dataset nkp37/OpenVid-1M \
--local-dir ./finetune/openvid \
--include "OpenVid_part3[0-9].zip"

huggingface-cli download --repo-type dataset nkp37/OpenVid-1M \
--local-dir ./finetune/openvid \
--include "OpenVid_part4[0-9].zip"
```

Then unzip into `./finetune/openvid/videos/`.

### Training 

```bash
# Download pretrained CogVideoX checkpoints
huggingface-cli download --repo-type model zai-org/CogVideoX-2b --local-dir ./ckpt/cogvideox-2b
huggingface-cli download --repo-type model zai-org/CogVideoX-5b --local-dir ./ckpt/cogvideox-5b

# Download pretrained vision encoder such as VideoMAEv2, VJEPA and put them into ./ckpt/. Such as ./ckpt/VideoMAEv2/vit_b_k710_dl_from_giant.pth

# Precompute video cache (shared for 2B/5B)
cd finetune/
bash scripts/dataset_precomputing.sh

# Training (adjust GPU count in scripts)
bash scripts/multigpu_VideoREPA_2B_sft.sh
bash scripts/multigpu_VideoREPA_5B_lora.sh
```

### Inference

Inference with the VideoREPA

```bash
# Transform checkpoint to diffuser format (only for sft)
# Put the scripts/merge.sh into the saved checkpoint-xxx/ and run:
bash merge.sh

# Then copy cogvideox-2b/ from ckpt/ to cogvideox-2b-infer/
# Delete the original transformer dir in cogvideox-2b-infer/
# Move the transformed transformer dir into it

# Modify model_index.config in cogvideox-2b-infer/
# "transformer": [
#   "models.cogvideox_align",
#   "CogVideoXTransformer3DModelAlign"
# ],

# Inference
cd inference/
bash scripts/infer_videorepa_2b_sft.sh
# bash scripts/infer_videorepa_5b_lora.sh
```

Or run inference directly with our released checkpoints.
Please download the weights from [Huggingface](https://huggingface.co/aHapBean/VideoREPA) and 

- For VideoREPA-5B, place `pytorch_lora_weights.safetensors` in `./inference/`

- For VideoREPA-2B, place the transformer directory inside `./ckpt/cogvideox-2b-infer/`

```bash
huggingface-cli download --repo-type model aHapBean/VideoREPA --local-dir ./
```

### Reproducing tips

We provide guidance for convenient results reproduction.

All experiments use **seed = 42** by default in our paper. However, note that randomness exists in both **video generation** and **VideoPhy evaluation**, so identical results across different devices (e.g., GPUs) may not be perfectly reproducible even with the same seed.

To reproduce demo videos, simply download the released [VideoREPA](https://huggingface.co/aHapBean/VideoREPA) checkpoints and run inference — similar videos can be generated using **VideoREPA-5B** (or **2B**).

To approximately reproduce the **VideoPhy scores**, you may either:
- Use the released evaluation videos, or  
- Run inference with the released checkpoints.

After the code release, we reproduced **VideoREPA-5B** on a **different device** and found differences in results due to randomness in the benchmark and generation process. Adjusting certain parameters such as `proj_coeff` (from **0.5 → 0.45**) helped restore the reported results, since the original settings were tuned with a different environment (device).

| Model | SA | PC |
|--------|------|------|
| VideoREPA-5B (reported) | 72.1 | 40.1 |
| VideoREPA-5B (reproduced) | 74.1 | 40.4 |

Changing the seed slightly may also help. It is expected that you can reproduce the performance trends without further parameter tuning.

## Contact

If you have any questions related to the code or the paper, feel free to email Xiangdong (`zhangxiangdong@sjtu.edu.cn`).

## Acknowledgement

This project is built upon and extends several distinguished open-source projects:

- [**CogVideo**](https://github.com/zai-org/CogVideo): A large-scale video generation framework developed by Tsinghua University, which provides the core architectural foundation for this work.  

- [**finetrainers**](https://github.com/huggingface/finetrainers): A high-efficiency training framework that helped enhance our fine-tuning pipeline.

- [**diffusers**](https://github.com/huggingface/diffusers): A go-to library for state-of-the-art pretrained diffusion models for generating images, audio, and even 3D structures of molecules.

## Citation

```
@article{zhang2025videorepa,
  title={VideoREPA: Learning Physics for Video Generation through Relational Alignment with Foundation Models},
  author={Zhang, Xiangdong and Liao, Jiaqi and Zhang, Shaofeng and Meng, Fanqing and Wan, Xiangpeng and Yan, Junchi and Cheng, Yu},
  journal={arXiv preprint arXiv:2505.23656},
  year={2025}
}
```

### More Generated Videos

<table border="0" style="width: 100%; text-align: center; margin-top: 1px;">
  <tr>
    <td><video src="https://github.com/user-attachments/assets/6ae1ef3f-5cf8-491b-87bb-5c53384ae74e" width="100%" controls autoplay loop muted></video></td>
    <td><video src="https://github.com/user-attachments/assets/de55cc3e-64ed-4961-bde4-ae84e1f47a93" width="100%" controls autoplay loop muted></video></td>
    <td><video src="https://github.com/user-attachments/assets/59a6a046-6b6d-4c1c-8f50-af12a943d9f3" width="100%" controls autoplay loop muted></video></td>
  </tr>
  <tr>    
    <td><video src="https://github.com/user-attachments/assets/a529ad95-b0d6-40f7-aa87-1c2e0c68a923" width="100%" controls autoplay loop muted></video></td>
    <td><video src="https://github.com/user-attachments/assets/d09520cf-d305-48b3-a8ab-c2619f343a84" width="100%" controls autoplay loop muted></video></td>
    <td><video src="https://github.com/user-attachments/assets/d86ee8a1-3448-4344-89d4-4f04f60bb3dd" width="100%" controls autoplay loop muted></video></td>
  </tr>
  <tr>
    <td><video src="https://github.com/user-attachments/assets/c377ad89-6324-486d-86fb-6489dec1d6af" width="100%" controls autoplay loop muted></video></td>
    <td><video src="https://github.com/user-attachments/assets/2d01754d-5b4d-4f13-9c75-298af07701dd" width="100%" controls autoplay loop muted></video></td>
    <td><video src="https://github.com/user-attachments/assets/88f5a3f1-a626-445e-a310-1f76753a6a74" width="100%" controls autoplay loop muted></video></td>
  </tr>

  <tr>
    <td><video src="https://github.com/user-attachments/assets/6fed6cc7-3b64-4821-aec3-8727a26f8a44" width="100%" controls autoplay loop muted></video></td>
    <td><video src="https://github.com/user-attachments/assets/240fa721-08b9-4168-8659-024472ea7155" width="100%" controls autoplay loop muted></video></td>
    <td><video src="https://github.com/user-attachments/assets/040e37f2-ee6c-47ee-8953-bd7814695226" width="100%" controls autoplay loop muted></video></td>
  </tr>

</table>
