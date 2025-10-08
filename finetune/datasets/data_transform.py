import cv2
import torch
import numpy as np
import decord
from decord import VideoReader
from torchvision import transforms
import os
# https://github1s.com/PKU-YuanGroup/Open-Sora-Plan/blob/main/opensora/dataset/transform.py#L43-L50

def _is_tensor_video_clip(clip):
    if not torch.is_tensor(clip):
        raise TypeError("clip should be Tensor. Got %s" % type(clip))

    if not clip.ndimension() == 4:
        raise ValueError("clip should be 4D. Got %dD" % clip.dim())

    return True

def crop(clip, i, j, h, w):
    """
    Args:
        clip (torch.tensor): Video clip to be cropped. Size is (T, C, H, W)
    """
    if len(clip.size()) != 4:
        raise ValueError("clip should be a 4D tensor")
    return clip[..., i: i + h, j: j + w]

def center_crop_th_tw(clip, th, tw, top_crop):
    if not _is_tensor_video_clip(clip):
        raise ValueError("clip should be a 4D torch.tensor")
    
    # import ipdb;ipdb.set_trace()
    h, w = clip.size(-2), clip.size(-1)
    tr = th / tw
    if h / w > tr:
        # hxw 720x1280  thxtw 320x640  hw_raito 9/16 > tr_ratio 8/16  newh=1280*320/640=640  neww=1280 
        new_h = int(w * tr)
        new_w = w
    else:
        # hxw 720x1280  thxtw 480x640  hw_raito 9/16 < tr_ratio 12/16   newh=720 neww=720/(12/16)=960  
        # hxw 1080x1920  thxtw 720x1280  hw_raito 9/16 = tr_ratio 9/16   newh=1080 neww=1080/(9/16)=1920  
        new_h = h
        new_w = int(h / tr)
    
    i = 0 if top_crop else int(round((h - new_h) / 2.0))
    j = int(round((w - new_w) / 2.0))
    return crop(clip, i, j, new_h, new_w)

def resize(clip, target_size, interpolation_mode):
    if len(target_size) != 2:
        raise ValueError(f"target size should be tuple (height, width), instead got {target_size}")
    return torch.nn.functional.interpolate(clip, size=target_size, mode=interpolation_mode, align_corners=True, antialias=True)


class CenterCropResizeVideo:
    '''
    First use the short side for cropping length,
    center crop video, then resize to the specified size
    '''

    def __init__(
            self,
            size,
            top_crop=False, 
            interpolation_mode="bilinear",
    ):
        if len(size) != 2:
            raise ValueError(f"size should be tuple (height, width), instead got {size}")
        self.size = size
        self.top_crop = top_crop
        self.interpolation_mode = interpolation_mode

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (T, C, H, W)
        Returns:
            torch.tensor: scale resized / center cropped video clip.
                size is (T, C, crop_size, crop_size)
        """
        clip_center_crop = center_crop_th_tw(clip, self.size[0], self.size[1], top_crop=self.top_crop)
        clip_center_crop_resize = resize(clip_center_crop, target_size=self.size,
                                         interpolation_mode=self.interpolation_mode)
        return clip_center_crop_resize

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size}, interpolation_mode={self.interpolation_mode}"


def process_video(input_video, output_video, frames_to_process=120, output_img_dir='imgs/'):
    # Initialize the VideoReader to read the video
    vr = VideoReader(input_video)
    
    transform = CenterCropResizeVideo(size=(160, 240))
    output_img_dir = 'imgs_openvid_16_24/'

    # Prepare the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, 30.0, (480, 720))

    # Create the directory for saving frames if it doesn't exist
    if not os.path.exists(output_img_dir):
        os.makedirs(output_img_dir)

    # Loop through the first 30 frames of the video
    for i in range(min(frames_to_process, len(vr))):
        # Get the frame as a numpy array
        frame = vr[i].asnumpy()  # shape: (height, width, channels)
        # Convert the frame to a torch tensor (CHW format)
        frame_tensor = torch.tensor(frame).permute(2, 0, 1).unsqueeze(0).float()    # float NOTE  # shape: (1, C, H, W)
        
        print(frame_tensor)
        # Apply the transformation (crop and resize)
        transformed_frame = transform(frame_tensor)
        # print(transformed_frame)
        print(f'transformed: {transformed_frame}')
        # exit()
        
        # Normalize back to the range [0, 255]
        transformed_frame = transformed_frame.squeeze(0).permute(1, 2, 0).byte().numpy()
        
        # Ensure that the values are clamped to the [0, 255] range
        transformed_frame = np.clip(transformed_frame, 0, 255)
        # Convert RGB (from Decord) to BGR (for OpenCV) NOTE here
        transformed_frame = cv2.cvtColor(transformed_frame, cv2.COLOR_RGB2BGR)

        # Write the frame to video
        out.write(transformed_frame)

        # Save the frame as an individual image
        img_filename = os.path.join(output_img_dir, f"frame_{i+1:03d}.jpg")
        cv2.imwrite(img_filename, transformed_frame)

    # Release the video writer
    out.release()
    print(f"Processed video saved to {output_video}")
    print(f"Frames saved as images to {output_img_dir}")

if __name__ == "__main__":
    input_video = "celebv___f2KtcXAxI_0.mp4"
    output_video = "cropped.mp4"
    process_video(input_video, output_video)