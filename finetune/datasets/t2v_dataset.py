import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import torch
from accelerate.logging import get_logger
from safetensors.torch import load_file, save_file
from torch.utils.data import Dataset
from torchvision import transforms
from typing_extensions import override
import pandas as pd
from finetune.constants import LOG_LEVEL, LOG_NAME
from .data_transform import CenterCropResizeVideo
from .utils import load_prompts, load_videos, preprocess_video_with_buckets, preprocess_video_with_resize, preprocess_video_with_resize_with_frame_idxs
import os
from torchvision.transforms import Normalize
if TYPE_CHECKING:
    from finetune.trainer import Trainer
import random
import time
import signal
# Must import after torch because this can sometimes lead to a nasty segmentation fault, or stack smashing error
# Very few bug reports but it happens. Look in decord Github issues for more relevant information.
import decord  # isort:skip

decord.bridge.set_bridge("torch")

logger = get_logger(LOG_NAME, LOG_LEVEL)
from multiprocessing import Process, Queue

class BaseT2VDataset(Dataset):
    """
    Base dataset class for Text-to-Video (T2V) training.

    This dataset loads prompts and videos for T2V training.

    Args:
        data_root (str): Root directory containing the dataset files
        caption_column (str): Path to file containing text prompts/captions
        video_column (str): Path to file containing video paths
        device (torch.device): Device to load the data on
        encode_video_fn (Callable[[torch.Tensor], torch.Tensor], optional): Function to encode videos
    """

    def __init__(
        self,
        max_num_frames: int,
        data_root: str,
        caption_column: str,
        video_column: str,
        precomputing: bool,
        device: torch.device = None,
        trainer: "Trainer" = None,
        train_data_path: str = None, 
        model_path: str = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()

        data_root = Path(data_root)
        base_path = base_path = data_root / "openvid/videos"
        assert train_data_path is not None, 'currently only openvid csv supported'
        self.train_data_path = train_data_path
        assert model_path is not None
        self.model_path = model_path # FIXME temporally used for resue the caching precomputing feature computed using vae scaling factor of CogVideoX-2B for 5B
        
        if train_data_path is not None:
            self.prompts = []
            self.videos = []
            self.frames_idxs = []            
            for one_train_data_path in train_data_path:
                df = pd.read_csv(one_train_data_path)
                self.prompts.extend(df['caption'].tolist())
                
                for index, row in df.iterrows():
                    self.videos.append(base_path / row['video'])
                    self.frames_idxs.append([None, None])
        else:
            self.prompts = load_prompts(data_root / caption_column)
            self.videos = load_videos(data_root / video_column)
            self.frames_idxs = None
        
        if precomputing:
            # To allow parallel pre-processing with different GPUs (processes)
            combined = list(zip(self.prompts, self.videos, self.frames_idxs))
            random.seed(time.time())
            random.shuffle(combined)
            self.prompts, self.videos, self.frames_idxs = zip(*combined)
            self.prompts = list(self.prompts)
            self.videos = list(self.videos)
            self.frames_idxs = list(self.frames_idxs)
        else:
            pass
        
        self.device = device
        self.encode_video = trainer.encode_video
        self.encode_text = trainer.encode_text
        self.trainer = trainer
        self.max_num_frames = max_num_frames

        # Check if all video files exist
        if any(not path.is_file() for path in self.videos):
            raise ValueError(
                f"Some video files were not found. Please ensure that all video files exist in the dataset directory. Missing file: {next(path for path in self.videos if not path.is_file())}"
            )

        # Check if number of prompts matches number of videos
        if len(self.videos) != len(self.prompts):
            raise ValueError(
                f"Expected length of prompts and videos to be the same but found {len(self.prompts)=} and {len(self.videos)=}. Please ensure that the number of caption prompts and videos match in your dataset."
            )

    def __len__(self) -> int:
        return len(self.videos)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if isinstance(index, list):
            # Here, index is actually a list of data objects that we need to return.
            # The BucketSampler should ideally return indices. But, in the sampler, we'd like
            # to have information about num_frames, height and width. Since this is not stored
            # as metadata, we need to read the video to get this information. You could read this
            # information without loading the full video in memory, but we do it anyway. In order
            # to not load the video twice (once to get the metadata, and once to return the loaded video
            # based on sampled indices), we cache it in the BucketSampler. When the sampler is
            # to yield, we yield the cache data instead of indices. So, this special check ensures
            # that data is not loaded a second time. PRs are welcome for improvements.
            return index

        prompt = self.prompts[index]
        video = self.videos[index]
        train_resolution_str = "x".join(str(x) for x in self.trainer.args.train_resolution)
        
        
        if len(self.train_data_path) == 1:
            cache_dir = self.trainer.args.data_root / "cache" / "openvid"
        else:
            raise NotImplementedError 
                
        if '-align' in self.trainer.args.model_name:
            model_name_tmp = self.trainer.args.model_name.replace("-align", "")
            video_latent_dir = cache_dir / "video_latent" / model_name_tmp / train_resolution_str
            frame_idx_dir = cache_dir / "frame_idx" / model_name_tmp / train_resolution_str
        else:
            video_latent_dir = cache_dir / "video_latent" / self.trainer.args.model_name / train_resolution_str
            frame_idx_dir = cache_dir / "frame_idx" / self.trainer.args.model_name / train_resolution_str
        
        prompt_embeddings_dir = cache_dir / "prompt_embeddings"
        
        video_latent_dir.mkdir(parents=True, exist_ok=True)
        prompt_embeddings_dir.mkdir(parents=True, exist_ok=True)
        frame_idx_dir.mkdir(parents=True, exist_ok=True)

        prompt_hash = str(hashlib.sha256(prompt.encode()).hexdigest())
        prompt_embedding_path = prompt_embeddings_dir / (prompt_hash + ".safetensors")
        
        
        encoded_video_path = video_latent_dir / (video.stem + f".safetensors")
        encoded_frame_idx_path = frame_idx_dir / (video.stem + f".safetensors")

        if prompt_embedding_path.exists():
            prompt_embedding = load_file(prompt_embedding_path)["prompt_embedding"]
            logger.debug(
                f"process {self.trainer.accelerator.process_index}: Loaded prompt embedding from {prompt_embedding_path}",
                main_process_only=False,
            )
        else:
            prompt_embedding = self.encode_text(prompt)
            prompt_embedding = prompt_embedding.to("cpu")
            # [1, seq_len, hidden_size] -> [seq_len, hidden_size]
            prompt_embedding = prompt_embedding[0]
            save_file({"prompt_embedding": prompt_embedding}, prompt_embedding_path)
            logger.info(f"Saved prompt embedding to {prompt_embedding_path}", main_process_only=False)
            

        if encoded_video_path.exists() and encoded_frame_idx_path.exists():
            encoded_video = load_file(encoded_video_path)["encoded_video"]
            
            if 'cogvideox-5b' in str(self.model_path):
                encoded_video = encoded_video / 1.15258426 * 0.7    # the vae scaling factors of 2B's and 5B's are different 
            else:
                assert 'cogvideox-2b' in str(self.model_path) 
            
            logger.debug(f"Loaded encoded video from {encoded_video_path}", main_process_only=False)
            video_reader = decord.VideoReader(uri=video.as_posix())
            frame_idx_list = load_file(encoded_frame_idx_path)["frame_idx_list"].tolist()
            frames = video_reader.get_batch(frame_idx_list) # F, H, W, C
                    
            frames = frames[:self.max_num_frames].float()   
            frames = frames.permute(0, 3, 1, 2).contiguous()
            frames = self.video_transform(frames)   # Reshape and center crop
            
            videomae_frames = frames
            videomae_frames = videomae_frames.permute(1, 0, 2, 3).contiguous() # C, F, H, W
        else:
            print(f'Pre-processing: {video}')
            assert 'cogvideox-2b' in str(self.model_path), 'when precomputing, the vae config should be cogvideox-2b'
            if self.frames_idxs is None:
                frames, frame_idx_list = self.preprocess(video)
            else:
                frames, frame_idx_list = self.preprocess_with_frame_idxs(video, self.frames_idxs[index])            

            frames = frames.to(self.device)
            frames = self.video_transform(frames)   # The input should be F, C, H, W
            # Convert to [B, C, F, H, W]
            frames = frames.unsqueeze(0)
            frames = frames.permute(0, 2, 1, 3, 4).contiguous()
            
            encoded_video = self.encode_video(frames)
            frames = frames.squeeze(0)
            videomae_frames = None
            frames = frames.to("cpu")  
            # [1, C, F, H, W] -> [C, F, H, W]
            encoded_video = encoded_video[0]
            encoded_video = encoded_video.to("cpu")
            save_file({"encoded_video": encoded_video}, encoded_video_path)
            save_file({"frame_idx_list": torch.tensor(frame_idx_list)}, encoded_frame_idx_path)
            
            logger.info(f"Saved encoded video to {encoded_video_path}", main_process_only=False)

        return {
            "prompt_embedding": prompt_embedding,
            "encoded_video": encoded_video,
            "video_metadata": {
                "num_frames": encoded_video.shape[1],
                "height": encoded_video.shape[2],
                "width": encoded_video.shape[3],
            },
            "raw_frames": videomae_frames,  # C, F, H, W
        }

    def preprocess(self, video_path: Path) -> torch.Tensor:
        """
        Loads and preprocesses a video.

        Args:
            video_path: Path to the video file to load.

        Returns:
            torch.Tensor: Video tensor of shape [F, C, H, W] where:
                - F is number of frames
                - C is number of channels (3 for RGB)
                - H is height
                - W is width
        """
        raise NotImplementedError("Subclass must implement this method")

    def preprocess_with_frame_idxs(self, video_path: Path, frame_idxs: list) -> torch.Tensor:
        """
        Loads and preprocesses a video.

        Args:
            video_path: Path to the video file to load.

        Returns:
            torch.Tensor: Video tensor of shape [F, C, H, W] where:
                - F is number of frames
                - C is number of channels (3 for RGB)
                - H is height
                - W is width
        """
        raise NotImplementedError("Subclass must implement this method")
    

    def video_transform(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Applies transformations to a video.

        Args:
            frames (torch.Tensor): A 4D tensor representing a video
                with shape [F, C, H, W] where:
                - F is number of frames
                - C is number of channels (3 for RGB)
                - H is height
                - W is width

        Returns:
            torch.Tensor: The transformed video tensor with the same shape as the input
        """
        raise NotImplementedError("Subclass must implement this method")


class T2VDatasetWithResize(BaseT2VDataset):
    """
    A dataset class for text-to-video generation that resizes inputs to fixed dimensions.

    This class preprocesses videos by resizing them to specified dimensions:
    - Videos are resized to max_num_frames x height x width

    Args:
        max_num_frames (int): Maximum number of frames to extract from videos
        height (int): Target height for resizing videos
        width (int): Target width for resizing videos
    """

    def __init__(self, height: int, width: int, *args, **kwargs) -> None:   
        super().__init__(*args, **kwargs)

        self.height = height
        self.width = width
        resize = CenterCropResizeVideo((self.height, self.width))

        self.__frame_transform = transforms.Compose([
            resize,
            transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)]
        )

    @override
    def preprocess(self, video_path: Path) -> torch.Tensor:
        return preprocess_video_with_resize(     
            video_path,
            self.max_num_frames,
            self.height,
            self.width,
            resize=False,   
        )

    def preprocess_with_frame_idxs(self, video_path: Path, frame_idxs: list) -> torch.Tensor:
        return preprocess_video_with_resize_with_frame_idxs(       
            video_path,
            self.max_num_frames,
            self.height,
            self.width,
            frame_idxs,
            resize=False,
        )

    @override
    def video_transform(self, frames: torch.Tensor) -> torch.Tensor:   
        return self.__frame_transform(frames)


class T2VDatasetWithBuckets(BaseT2VDataset):
    def __init__(
        self,
        video_resolution_buckets: List[Tuple[int, int, int]],
        vae_temporal_compression_ratio: int,
        vae_height_compression_ratio: int,
        vae_width_compression_ratio: int,
        *args,
        **kwargs,
    ) -> None:
        """ """
        super().__init__(*args, **kwargs)

        self.video_resolution_buckets = [
            (
                int(b[0] / vae_temporal_compression_ratio),
                int(b[1] / vae_height_compression_ratio),
                int(b[2] / vae_width_compression_ratio),
            )
            for b in video_resolution_buckets
        ]

        self.__frame_transform = transforms.Compose([transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)])

    @override
    def preprocess(self, video_path: Path) -> torch.Tensor:
        return preprocess_video_with_buckets(video_path, self.video_resolution_buckets)

    @override
    def video_transform(self, frames: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.__frame_transform(f) for f in frames], dim=0)
