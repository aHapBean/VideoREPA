# VideoPhy

Thanks for the great work [VideoPhy](https://github.com/Hritikbansal/videophy) from [Hritik Bansal](https://github.com/Hritikbansal).

We provide brief guidance on how to use this repo to evaluate VideoREPA.

## 1. Create conda environment

```bash
conda create -n videophy python=3.10
conda activate videophy
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Download evaluation model checkpoint

The model checkpoint is publicly available on [🤗 Model Hub](https://huggingface.co/videophysics/videocon_physics/tree/main).

```bash
git lfs install
git clone https://huggingface.co/videophysics/videocon_physics
```

## 4. Evaluation

```bash
# Before running generate_csv.py, set `target_ver` and `prefix`
# to the directory containing your generated 344 VideoPhy videos
python generate_csv.py

# Replace the checkpoint path in eval_pipeline.sh with the downloaded model and modify the `INPUT_CSV`
bash eval_pipeline.sh
```
