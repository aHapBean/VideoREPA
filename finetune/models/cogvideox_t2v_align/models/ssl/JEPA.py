import finetune.models.cogvideox_t2v_align.models.ssl.vjepa_vision_transformer as vit
import torch

def init_model(
    device,
    pretrained,
    model_name,
    patch_size=16,
    crop_size=224,
    # Video specific parameters
    frames_per_clip=16,
    tubelet_size=2,
    use_sdpa=False,
    use_SiLU=False,
    tight_SiLU=True,
    uniform_power=False,
    checkpoint_key='target_encoder'
):
    encoder = vit.__dict__[model_name](
        img_size=crop_size,
        patch_size=patch_size,
        num_frames=frames_per_clip,
        tubelet_size=tubelet_size,
        uniform_power=uniform_power,
        use_sdpa=use_sdpa,
        use_SiLU=use_SiLU,
        tight_SiLU=tight_SiLU,
    )
    print(encoder)
    encoder.to(device)
    # if pretrained is None:
    #     return encoder
    assert pretrained is not None
    encoder = load_pretrained(encoder=encoder, pretrained=pretrained, checkpoint_key=checkpoint_key)
    return encoder



def load_pretrained(
    encoder,
    pretrained,
    checkpoint_key='target_encoder'
):
    print(f'Loading pretrained model from {pretrained}')
    checkpoint = torch.load(pretrained, map_location='cpu')
    try:
        pretrained_dict = checkpoint[checkpoint_key]
    except Exception:
        pretrained_dict = checkpoint['encoder']

    pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}
    pretrained_dict = {k.replace('backbone.', ''): v for k, v in pretrained_dict.items()}
    for k, v in encoder.state_dict().items():
        if k not in pretrained_dict:
            print(f'key "{k}" could not be found in loaded state dict')
        elif pretrained_dict[k].shape != v.shape:
            print(f'key "{k}" is of different shape in model and loaded state dict')
            pretrained_dict[k] = v
    msg = encoder.load_state_dict(pretrained_dict, strict=False)
    print(encoder)
    print(f'loaded pretrained model with msg: {msg}')
    print(f'loaded pretrained encoder from epoch: {checkpoint["epoch"]}\n path: {pretrained}')
    del checkpoint
    return encoder

def load_VJEPA(device, pretrained_path='/mnt/zhangxiangdong/VideoGeneration/CogVideo/ckpt/vjepa_l'):
    encoder = init_model(
        crop_size=224,
        device=device,
        pretrained=pretrained_path,
        model_name='vit_large',
        patch_size=16,
        tubelet_size=2,
        frames_per_clip=16,
        uniform_power=True,
        checkpoint_key='target_encoder',
        use_SiLU=False,
        tight_SiLU=False,
        use_sdpa=True,
    )
    return encoder 