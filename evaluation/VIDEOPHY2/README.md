# VideoPhy

Thanks for the great work [VideoPhy2](https://github.com/Hritikbansal/videophy) from [Hritik Bansal](https://github.com/Hritikbansal).

We provide brief guidance on how to use this repo to evaluate VideoREPA.

## 1. Installation

1. Follow the same instructions mentioned in the VideoPhy-1 [README](release/videophy/README.md).

2. AutoEval: The model checkpoint is publicly available on [🤗 Model](https://huggingface.co/hbXNov/videophy_2_auto/tree/main).

## 2. Evaluation

```bash
# Before running generate_csv.py, set `target_ver` and `prefix`
# to the directory containing your generated 344 VideoPhy videos
python generate_csv.py

# Replace the checkpoint path in eval_pipeline.sh with the downloaded model and modify the `INPUT_CSV`
bash eval_pipeline.sh
```
