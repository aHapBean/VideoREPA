from typing import Any, Dict, List, Tuple

import torch
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
    CogVideoXDDIMScheduler,
    # CogVideoXPipeline,
    # CogVideoXTransformer3DModel,
)
from finetune.models.cogvideox_t2v_align.models.cogvideox_align import CogVideoXTransformer3DModelAlign, CogVideoXPipelineAlign
from finetune.models.cogvideox_t2v_align.models.ssl.VideoMAEv2 import vit_base_patch16_224, vit_large_patch16_224, vit_huge_patch16_224, vit_giant_patch14_224
from finetune.models.cogvideox_t2v_align.models.ssl.VideoMAE import vit_base_patch16_224 as VideoMAE_vit_base_patch16_224

from diffusers.models.embeddings import get_3d_rotary_pos_embed
from PIL import Image
from transformers import AutoTokenizer, T5EncoderModel
from typing_extensions import override

from finetune.schemas import Components
from finetune.trainer import Trainer
from finetune.utils import unwrap_model
import torch.nn as nn
from ..utils import register
from torchvision.transforms import Normalize
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

class CogVideoXT2VAlignLoraTrainer(Trainer):
    UNLOAD_LIST = ["text_encoder", "vae"]

    def initialize_vision_encoder(self):
        assert len(self.args.align_models) == 1, 'Currently support one alignment model'
        if self.args.align_models[0] == "VideoMAEv2":
            self.vision_encoder = vit_base_patch16_224().to(self.accelerator.device)
            self.vision_encoder.from_pretrained('../ckpt/VideoMAEv2/vit_b_k710_dl_from_giant.pth')  # The from pretrained return None
            # freeze the parameter
            self.vision_encoder.eval()
            # Actually no need to set False because it is not going through the optimizer
            for param in self.vision_encoder.parameters():  
                param.require_grad = False
        elif self.args.align_models[0] == "VideoMAE":
            self.vision_encoder = VideoMAE_vit_base_patch16_224().to(self.accelerator.device)
            self.vision_encoder.from_pretrained('../ckpt/VideoMAE/k400_videomae_pretrain_base_patch16_224_frame_16x4_tube_mask_ratio_0_9_e1600.pth')
            self.vision_encoder.eval()
            for param in self.vision_encoder.parameters():
                param.require_grad = False
        elif self.args.align_models[0] == 'OminiMAE':
            from finetune.models.cogvideox_t2v_align.models.ssl.omini_mae import vit_base_mae_pretraining
            self.vision_encoder = vit_base_mae_pretraining().to(self.accelerator.device)
            self.vision_encoder.eval()
            for param in self.vision_encoder.parameters():
                param.require_grad = False          
            self.vision_encoder.tubelet_size = 2
            self.vision_encoder.patch_size = 16
            self.vision_encoder.embed_dim = 768    
        elif self.args.align_models[0] == 'DINOv2':
            self.vision_encoder = torch.hub.load('facebookresearch/dinov2', f'dinov2_vitb14').to(self.accelerator.device)
            self.vision_encoder.eval()
            del self.vision_encoder.head
            self.vision_encoder.head = torch.nn.Identity()
            for param in self.vision_encoder.parameters():
                param.require_grad = False 
        elif self.args.align_models[0] == 'VJEPA':
            from finetune.models.cogvideox_t2v_align.models.ssl.JEPA import load_VJEPA
            self.vision_encoder = load_VJEPA(device=self.accelerator.device, pretrained_path='../ckpt/vjepa_l/vitl16.pth.tar')
            self.vision_encoder.eval()
            for param in self.vision_encoder.parameters():
                param.require_grad = False 
        elif self.args.align_models[0] == "VJEPA2":
            self.vision_encoder, _ = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_vit_large')
            self.vision_encoder = self.vision_encoder.to(self.accelerator.device)
            self.vision_encoder.eval()
            del self.vision_encoder.norm
            self.vision_encoder.norm = torch.nn.Identity()
            for param in self.vision_encoder.parameters():
                param.require_grad = False                       
        else:
            raise NotImplementedError

    @override
    def load_components(self) -> Components:
        components = Components()
        model_path = str(self.args.model_path)

        components.pipeline_cls = CogVideoXPipelineAlign

        components.tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")

        components.text_encoder = T5EncoderModel.from_pretrained(model_path, subfolder="text_encoder")

        # add interface for the parameter of feature alignment
        components.transformer = CogVideoXTransformer3DModelAlign.from_pretrained(model_path, subfolder="transformer", align_layer=self.args.align_layer, align_dims=self.args.align_dims, projector_dim=self.args.projector_dim)  
        
        components.vae = AutoencoderKLCogVideoX.from_pretrained(model_path, subfolder="vae")

        components.scheduler = CogVideoXDPMScheduler.from_pretrained(model_path, subfolder="scheduler")
        # components.scheduler = CogVideoXDDIMScheduler.from_pretrained(model_path, subfolder="scheduler")

        return components

    @override
    def initialize_pipeline(self) -> CogVideoXPipelineAlign:
        pipe = CogVideoXPipelineAlign(
            tokenizer=self.components.tokenizer,
            text_encoder=self.components.text_encoder,
            vae=self.components.vae,
            transformer=unwrap_model(self.accelerator, self.components.transformer),
            scheduler=self.components.scheduler,
        )
        return pipe

    @override
    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        # This is used in the dataloader
        # shape of input video: [B, C, F, H, W]
        vae = self.components.vae
        video = video.to(vae.device, dtype=vae.dtype)
        latent_dist = vae.encode(video).latent_dist
        latent = latent_dist.sample() * vae.config.scaling_factor
        return latent

    # def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
    #     latents = latents.permute(0, 2, 1, 3, 4)  # [batch_size, num_channels, num_frames, height, width]
    #     latents = 1 / self.vae_scaling_factor_image * latents

    #     frames = self.vae.decode(latents).sample
    #     return frames

    @override
    def encode_text(self, prompt: str) -> torch.Tensor:
        prompt_token_ids = self.components.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.state.transformer_config.max_text_seq_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        prompt_token_ids = prompt_token_ids.input_ids
        prompt_embedding = self.components.text_encoder(prompt_token_ids.to(self.accelerator.device))[0]
        return prompt_embedding

    @override
    def collate_fn(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        ret = {"encoded_videos": [], "prompt_embedding": [], "raw_frames": []}

        for sample in samples:
            encoded_video = sample["encoded_video"]
            prompt_embedding = sample["prompt_embedding"]
            raw_frames = sample["raw_frames"]

            ret["encoded_videos"].append(encoded_video)
            ret["prompt_embedding"].append(prompt_embedding)
            ret["raw_frames"].append(raw_frames)

        ret["encoded_videos"] = torch.stack(ret["encoded_videos"])
        ret["prompt_embedding"] = torch.stack(ret["prompt_embedding"])
        ret["raw_frames"] = torch.stack(ret["raw_frames"])

        return ret

    @override
    def compute_loss(self, batch) -> torch.Tensor:
        prompt_embedding = batch["prompt_embedding"]
        latent = batch["encoded_videos"]
        raw_frames = batch["raw_frames"]    # [B, C, F, H, W] whose value range from -1 to 1, e.g. torch.Size([Batch_size, 3, 49, 480, 720])
        
        # pre-process for vision encoder
        B, C, F, H, W = raw_frames.shape 
        raw_frames = raw_frames.transpose(1, 2).flatten(0, 1)   # B * F, C, H, W
        if self.args.align_models[0] in ['VideoMAEv2', 'VideoMAE', 'OminiMAE', "VJEPA", "VJEPA2"]:
            raw_frames = (raw_frames + 1.0) / 2.0
            raw_frames = Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(raw_frames)    # should be NCHW
        elif self.args.align_models[0] == 'DINOv2':
            raw_frames = (raw_frames + 1.0) / 2.0
            raw_frames = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(raw_frames)
        else:
            raise NotImplementedError
        raw_frames = raw_frames.reshape(B, F, C, H, W).transpose(1, 2)
        
        
        # pre-process frames for Video Foundation Models
        assert len(self.args.align_models) == 1, "Support only align one model currently"
        if self.args.align_models[0] in ['VideoMAEv2', 'VJEPA', 'VJEPA2', 'VideoMAE', 'OminiMAE']:
            repa_raw_frames = raw_frames[:, :, 1:]  # remove the first frames
            B, C, F, H, W = repa_raw_frames.shape 
            
            repa_raw_frames = repa_raw_frames.transpose(1, 2).flatten(0, 1)
            # 480x720 -> 160x240
            repa_raw_frames = torch.nn.functional.interpolate(repa_raw_frames, (H // 3, W // 3), mode='bicubic')    # hard coded
            repa_raw_frames = repa_raw_frames.reshape(B, F, C, H // 3, W // 3).transpose(1, 2)  # B, C, F, H, W
        elif self.args.align_models[0] == 'DINOv2':
            repa_raw_frames = raw_frames
            B, C, F, H, W = repa_raw_frames.shape 
            repa_raw_frames = repa_raw_frames.transpose(1, 2).flatten(0, 1) # B * F, C, H, W
            input_resolution = (420, 630)   # to fit the patch size 14 in DINOv2
            repa_raw_frames = torch.nn.functional.interpolate(repa_raw_frames, input_resolution, mode='bicubic')
            repa_raw_frames = repa_raw_frames.reshape(B, F, C, input_resolution[0], input_resolution[1]).transpose(1, 2)  # B, C, F, H, W
        
        
        # encode the frames with vision encoders
        with torch.no_grad():
            if self.args.align_models[0] in ['VideoMAEv2', 'VJEPA', 'VJEPA2', 'VideoMAE', 'OminiMAE']:
                B, C, F, H, W = repa_raw_frames.shape
                # encoding the frames with vision encoder: B, 3, 48, 160, 240 -> B, 24x10x15, C
                align_target = self.vision_encoder(repa_raw_frames)
                # B, 24x10x15, D -> B, D, 24, 10, 15
                align_target = align_target.transpose(1, 2).reshape(B, -1, F // self.vision_encoder.tubelet_size, H // self.vision_encoder.patch_size, W // self.vision_encoder.patch_size)
            elif self.args.align_models[0] == 'DINOv2':
                B, C, F, H, W = repa_raw_frames.shape
                repa_raw_frames = repa_raw_frames.transpose(1, 2).flatten(0, 1)
                group_size = 128  # 32 / 64 / 128 to avoid OOM
                chunked = repa_raw_frames.chunk((B * F) // group_size, dim=0)
                
                features = []
                for frames in chunked:
                    group, C, H, W = frames.shape
                    output = self.vision_encoder.forward_features(frames)['x_norm_patchtokens'].reshape(group, input_resolution[0] // self.vision_encoder.patch_size, input_resolution[1] // self.vision_encoder.patch_size, self.vision_encoder.embed_dim)
                    features.append(output)
                features = torch.cat(features, dim=0)
                features = features.reshape(B, F, input_resolution[0] // self.vision_encoder.patch_size, input_resolution[1] // self.vision_encoder.patch_size, self.vision_encoder.embed_dim)
        
        align_targets = []
        if self.args.align_models[0] in ['VideoMAEv2', 'VJEPA', 'VJEPA2', 'VideoMAE', 'OminiMAE']:
            align_target = align_target.flatten(2).transpose(1, 2)  # B, 24x10x15, C
            align_targets.append(align_target)        
        elif self.args.align_models[0] == 'DINOv2':
            first_frame_feature = features[:, :1].permute(0, 4, 1, 2, 3)   # B, 1, H, W, C -> B, C, 1, H, W
            features = features[:, 1:]
            B, F, H, W, C = features.shape
            align_target = features.permute(0, 2, 3, 4, 1).flatten(0, 2)
            # To align with the features from CogVideoX, the encoded features are avg pooled to 1/4
            align_target = torch.nn.functional.avg_pool1d(align_target, kernel_size=4, stride=4)
            align_target = align_target.reshape(B, H, W, C, F // 4).permute(0, 3, 4, 1, 2)
            align_target = torch.cat([first_frame_feature, align_target], dim=2)
            align_target = align_target.flatten(2).transpose(1, 2)  # B, 13x30x45, C
            align_targets.append(align_target)  
            

        patch_size_t = self.state.transformer_config.patch_size_t
        if patch_size_t is not None:
            raise NotImplementedError("This is for CogVideoX1.5 but the 1.5 is not used in VideoREPA")
            ncopy = latent.shape[2] % patch_size_t
            # Copy the first frame ncopy times to match patch_size_t
            first_frame = latent[:, :, :1, :, :]
            latent = torch.cat([first_frame.repeat(1, 1, ncopy, 1, 1), latent], dim=2)
            assert latent.shape[2] % patch_size_t == 0

        batch_size, num_channels, num_frames, height, width = latent.shape

        # Get prompt embeddings
        _, seq_len, _ = prompt_embedding.shape
        prompt_embedding = prompt_embedding.view(batch_size, seq_len, -1).to(dtype=latent.dtype)

        # Sample a random timestep for each sample
        timesteps = torch.randint(
            0, self.components.scheduler.config.num_train_timesteps, (batch_size,), device=self.accelerator.device
        )
        timesteps = timesteps.long()

        # Add noise to latent
        latent = latent.permute(0, 2, 1, 3, 4)  # from [B, C, F, H, W] to [B, F, C, H, W]
        noise = torch.randn_like(latent)
        latent_added_noise = self.components.scheduler.add_noise(latent, noise, timesteps)

        # Prepare rotary embeds
        vae_scale_factor_spatial = 2 ** (len(self.components.vae.config.block_out_channels) - 1)
        transformer_config = self.state.transformer_config
        rotary_emb = (
            self.prepare_rotary_positional_embeddings(
                height=height * vae_scale_factor_spatial,
                width=width * vae_scale_factor_spatial,
                num_frames=num_frames,
                transformer_config=transformer_config,
                vae_scale_factor_spatial=vae_scale_factor_spatial,
                device=self.accelerator.device,
            )
            if transformer_config.use_rotary_positional_embeddings
            else None
        )
        
        # Predict noise
        predicted_noises, aligns = self.components.transformer(
            hidden_states=latent_added_noise,
            encoder_hidden_states=prompt_embedding,
            timestep=timesteps,
            image_rotary_emb=rotary_emb,
            return_dict=False,
        )
        predicted_noise = predicted_noises[0]
        
        # Aligning features from CogVideoX to pre-trained frozen vision encoders
        align = aligns[0]
        align = align.reshape(B, 13, 60 // 2, 90 // 2, -1)  # TODO: remove hard coded
        if self.args.align_models[0] in ['VideoMAEv2', 'VJEPA', 'VJEPA2', 'VideoMAE', 'OminiMAE']:
            # remove the first frame
            align = align[:, 1:]    
        aligns = [align]
        if self.args.align_models[0] == 'DINOv2':
            # Only able to perform REPA loss when using DINOv2
            assert self.args.loss == 'cosine_similarity'
        
        if self.args.loss == 'cosine_similarity':
            # REPA loss
            proj_loss = 0
            align = aligns[0].permute(0, 4, 1, 2, 3)    # B, C, F, H, W
            if self.args.align_models[0] != "DINOv2": 
                align = torch.nn.functional.interpolate(align, scale_factor=(2.0, 1.0, 1.0), mode='trilinear') 
            
            if self.args.align_models[0] in ['VideoMAEv2', 'VJEPA', 'VJEPA2', 'VideoMAE', 'OminiMAE']:
                B, C, F, H, W = align.shape
                align = align.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W)
                align = self.components.transformer.downsampler_cogvideo_output(align.to(torch.bfloat16))   # 30x45 -> 10x15
                align = align.reshape(B, F, C, H // 3, W // 3).permute(0, 2, 1, 3, 4)   # B, C, F, H, W
            
            align = align.flatten(2).transpose(1, 2).flatten(0, 1)  # BFHW, C
            align_target = align_targets[0].flatten(0, 1)
            align = torch.nn.functional.normalize(align, dim=-1) 
            align_target = torch.nn.functional.normalize(align_target, dim=-1) 
            assert align_target.shape[-1] == align.shape[-1] == self.args.align_dims[0]  # NOTE here

            proj_loss += (-(align_target * align)).sum(dim=-1).mean(dim=0) 
        
        elif self.args.loss == 'token_relation_distillation':
            # TRD loss in VideoREPA
            assert len(aligns) == 1
            align = aligns[0].permute(0, 4, 1, 2, 3)   # B, F, H, W, C -> B, C, F, H, W  (e.g. B, 768, 12, 30, 45)
            # upsample the temporal dimension (from 12 to 24) to match the dimension in VideoMAEv2
            align = torch.nn.functional.interpolate(align, scale_factor=(2.0, 1.0, 1.0), mode='trilinear')
    
            # downsample the representation of VDM
            B, C, F, H, W = align.shape
            align = align.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W)
            align = self.components.transformer.downsampler_cogvideo_output(align.to(torch.bfloat16))
            align = align.reshape(B, F, C, H // 3, W // 3)
            
            align = align.permute(0, 1, 3, 4, 2)   # B, F, H, W, C          
            token_relation_distillation_loss = 0
            align = align.flatten(2, 3) # B, F, H*W, C
            align_target = align_targets[0].reshape(B, F, 10 * 15, -1)  # B, 12, 10 * 15, D
            
            # normalize before calculate Gram matrix
            align = torch.nn.functional.normalize(align, dim=-1)
            align_target = torch.nn.functional.normalize(align_target, dim=-1)
            assert align.shape[-1] == align_target.shape[-1] == self.args.align_dims[0]

            # BF, HW, C @ BF, C, FHW -> BF, HW, FHW
            align_sim = torch.bmm(align.flatten(0, 1), align.flatten(1, 2).unsqueeze(1).expand(-1, F, -1, -1).flatten(0, 1).transpose(1, 2))
            align_target_sim = torch.bmm(align_target.flatten(0, 1), align_target.flatten(1, 2).unsqueeze(1).expand(-1, F, -1, -1).flatten(0, 1).transpose(1, 2))
            assert align_sim.shape == align_target_sim.shape
            # or refer to more concise implementation: B, FHW, C @ B, C, FHW -> B, FHW, FHW
            # align_sim = torch.bmm(align.flatten(1, 2), align.flatten(1, 2).transpose(1, 2))
            # align_target_sim = torch.bmm(align_target.flatten(1, 2), align_target.flatten(1, 2).transpose(1, 2))
            token_relation_distillation_loss = nn.functional.relu((align_sim - align_target_sim).abs() - self.args.margin).mean()

        elif self.args.loss == 'token_relation_distillation_only_spatial':
            # pre-process
            align = aligns[0].permute(0, 4, 1, 2, 3)
            align = torch.nn.functional.interpolate(align, scale_factor=(2.0, 1.0, 1.0), mode='trilinear') 
            B, C, F, H, W = align.shape
            align = align.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W)
            align = self.components.transformer.downsampler_cogvideo_output(align.to(torch.bfloat16))
            
            align = align.reshape(B, F, C, H // 3, W // 3)
            align = align.permute(0, 1, 3, 4, 2)

            # calculate loss
            token_relation_distillation_loss = 0
            align = align.flatten(2, 3)
            align_target = align_targets[0].reshape(B, F, 10 * 15, -1)
            align = torch.nn.functional.normalize(align, dim=-1)
            align_target = torch.nn.functional.normalize(align_target, dim=-1)
            
            assert align.shape[-1] == align_target.shape[-1] == self.args.align_dims[0]
            align_sim = torch.bmm(align.flatten(0, 1), align.flatten(0, 1).transpose(1, 2))
            align_target_sim = torch.bmm(align_target.flatten(0, 1), align_target.flatten(0, 1).transpose(1, 2)) 
            assert align_sim.shape == align_target_sim.shape
            token_relation_distillation_loss = nn.functional.relu((align_sim - align_target_sim).abs() - self.args.margin_matrix).mean()
        
        elif self.args.loss == 'token_relation_distillation_only_temporal':
            # pre-process
            align = aligns[0].permute(0, 4, 1, 2, 3)
            align = torch.nn.functional.interpolate(align, scale_factor=(2.0, 1.0, 1.0), mode='trilinear')  
                
            B, C, F, H, W = align.shape
            align = align.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W)   
            align = self.components.transformer.downsampler_cogvideo_output(align.to(torch.bfloat16))   
            
            align = align.reshape(B, F, C, H // 3, W // 3)
            align = align.permute(0, 1, 3, 4, 2)   

            token_relation_distillation_loss = 0
            align_temporal = align.flatten(2, 3) 
            align_target_temporal = align_targets[0].reshape(B, F, 10 * 15, -1)  
            align_temporal = torch.nn.functional.normalize(align_temporal, dim=-1)
            align_target_temporal = torch.nn.functional.normalize(align_target_temporal, dim=-1)
            
            assert align_temporal.shape[-1] == align_target_temporal.shape[-1] == self.args.align_dims[0]  
            align_sim = torch.bmm(align_temporal.flatten(1, 2), align_temporal.flatten(1, 2).transpose(1, 2))  
            align_target_temporal_sim = torch.bmm(align_target_temporal.flatten(1, 2), align_target_temporal.flatten(1, 2).transpose(1, 2)) 

            assert align_sim.shape == align_target_temporal_sim.shape
            token_relation_distillation_loss = nn.functional.relu((align_sim - align_target_temporal_sim).abs() - self.args.margin_matrix)
            
            token_relation_distillation_loss = token_relation_distillation_loss.clone()   # To prevent the following inplace operation which will raise gradient backward error

            token_relation_distillation_loss = token_relation_distillation_loss.reshape(B, 24, 10 * 15, 24, 10 * 15)
            for iddx in range(24):
                token_relation_distillation_loss[:, iddx, :, iddx, :] = torch.tensor(0.0)  
            token_relation_distillation_loss = token_relation_distillation_loss.mean() * (B * 24. * 10 * 15 * 24 * 10 * 15) / (B * 24. * 10 * 15 * (24 - 1) * 10 * 15)          

        else:
            raise NotImplementedError
        
        # Denoise
        latent_pred = self.components.scheduler.get_velocity(predicted_noise, latent_added_noise, timesteps)

        alphas_cumprod = self.components.scheduler.alphas_cumprod[timesteps]
        weights = 1 / (1 - alphas_cumprod)
        while len(weights.shape) < len(latent_pred.shape):
            weights = weights.unsqueeze(-1)

        loss = torch.mean((weights * (latent_pred - latent) ** 2).reshape(batch_size, -1), dim=1)
        loss = loss.mean()

        if self.args.loss == 'token_relation_distillation' or self.args.loss == 'token_relation_distillation_only_spatial' or self.args.loss == 'token_relation_distillation_only_temporal':
            return [loss, None, token_relation_distillation_loss]
        return [loss, proj_loss]

    @override
    def validation_step(
        self, eval_data: Dict[str, Any], pipe: CogVideoXPipelineAlign
    ) -> List[Tuple[str, Image.Image | List[Image.Image]]]:
        """
        Return the data that needs to be saved. For videos, the data format is List[PIL],
        and for images, the data format is PIL
            video_generate, list_of_frames_feature_maps = pipe(
                height=height,
                width=width,
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                num_inference_steps=num_inference_steps,
                num_frames=num_frames,
                use_dynamic_cfg=True,
                guidance_scale=guidance_scale,
                generator=torch.Generator().manual_seed(seed),
                feature_maps=True,
            )        
        """
        prompt, image, video = eval_data["prompt"], eval_data["image"], eval_data["video"]

        video_generate = pipe(
            num_frames=self.state.train_frames,
            height=self.state.train_height,
            width=self.state.train_width,
            prompt=prompt,
            generator=self.state.generator,
        ).frames[0]
        return [("video", video_generate)]
  

    def prepare_rotary_positional_embeddings(
        self,
        height: int,
        width: int,
        num_frames: int,
        transformer_config: Dict,
        vae_scale_factor_spatial: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        grid_height = height // (vae_scale_factor_spatial * transformer_config.patch_size)
        grid_width = width // (vae_scale_factor_spatial * transformer_config.patch_size)

        if transformer_config.patch_size_t is None:
            base_num_frames = num_frames
        else:
            base_num_frames = (num_frames + transformer_config.patch_size_t - 1) // transformer_config.patch_size_t
        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=transformer_config.attention_head_dim,
            crops_coords=None,
            grid_size=(grid_height, grid_width),
            temporal_size=base_num_frames,
            grid_type="slice",
            max_size=(grid_height, grid_width),
            device=device,
        )

        return freqs_cos, freqs_sin


register("cogvideox-t2v-align", "lora", CogVideoXT2VAlignLoraTrainer)
