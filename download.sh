# dataset
huggingface-cli download --repo-type dataset nkp37/OpenVid-1M \
--local-dir ./finetune/openvid \
--include "OpenVid_part3[0-9].zip"

huggingface-cli download --repo-type dataset nkp37/OpenVid-1M \
--local-dir ./finetune/openvid \
--include "OpenVid_part4[0-9].zip"

# model
huggingface-cli download --repo-type model zai-org/CogVideoX-2b \
--local-dir ./ckpt/cogvideox-2b

huggingface-cli download --repo-type model zai-org/CogVideoX-5b \
--local-dir ./ckpt/cogvideox-5b