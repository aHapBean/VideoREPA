from ..cogvideox_t2v_align.lora_trainer import CogVideoXT2VAlignLoraTrainer
from ..utils import register


class CogVideoXT2VAlignSftTrainer(CogVideoXT2VAlignLoraTrainer):
    pass


register("cogvideox-t2v-align", "sft", CogVideoXT2VAlignSftTrainer)
