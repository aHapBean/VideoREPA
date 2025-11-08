import pandas as pd
import os

target_ver = 'VideoREPA_2B'
prefix = f'/mnt/petrelfs/zhangxiangdong/VideoGeneration/CogVideo/inference/phygenbench/{target_ver}/videophy2/'

files = os.listdir(prefix)
print(len(files))   # 590 


data = []

for fl in files:
    assert fl.endswith('.mp4'), "The generation isn't over because the temporary directory exists"
    caption = ' '.join(fl.removesuffix('mp4').split('_'))
    
    video_path = prefix + fl
    
    data.append([video_path, caption])

df = pd.DataFrame(data, columns=['videopath', 'caption'])

output_file = f'./csv_file/{target_ver}.csv'
df.to_csv(output_file, index=False, encoding='utf-8')

print(f"Data has been written to {output_file}")
