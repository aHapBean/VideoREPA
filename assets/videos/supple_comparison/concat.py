from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.compositing.CompositeVideoClip import clips_array
import os

# --- 配置 ---
input_video_files = [
    "supp_cogvideox_01.mp4", "supp_cogvideox_02.mp4", "supp_cogvideox_03.mp4", "supp_cogvideox_04.mp4",
    "supp_videorepa_01.mp4", "supp_videorepa_02.mp4", "supp_videorepa_03.mp4", "supp_videorepa_04.mp4",
    "supp_cogvideox_05.mp4", "supp_cogvideox_06.mp4", "supp_cogvideox_07.mp4", "supp_cogvideox_08.mp4",
    "supp_videorepa_05.mp4", "supp_videorepa_06.mp4", "supp_videorepa_07.mp4", "supp_videorepa_08.mp4",
]

output_video_file = "output_grid_video.mp4"

# 网格布局 (4行, 4列)
rows = 4
cols = 4

# 目标尺寸，所有视频都会被缩放到这个尺寸
target_size = (720, 480) # 明确设置一个目标尺寸

# 输出视频的帧率 (FPS)
output_fps = 24

# 【已修正】为MP4容器选择兼容的编码器
codec = "libx264" # H.264编码器，与.mp4容器兼容

# --- 脚本开始 ---

if len(input_video_files) != rows * cols:
    print(f"错误: 输入文件数量 ({len(input_video_files)}) 与设定的网格 ({rows}x{cols}={rows*cols}) 不匹配。")
    exit()

for file in input_video_files:
    if not os.path.exists(file):
        print(f"错误: 文件 {file} 不存在。")
        exit()

print("正在加载和处理视频...")
video_clips = []
min_duration = float('inf') # 用于找到所有视频中的最短时长

for i, file in enumerate(input_video_files):
    print(f"处理视频: {file} ({i+1}/{len(input_video_files)})")
    try:
        clip = VideoFileClip(file)

        # 【已修正】确保缩放操作生效
        if clip.size != target_size:
             print(f"缩放视频 {file} 从 {clip.size} 到 {target_size}")
             # 将返回的新剪辑赋值回 clip 变量
             clip.resized(new_size=target_size)
        
        # 记录最短时长
        if clip.duration < min_duration:
            min_duration = clip.duration
            
        video_clips.append(clip)
    except Exception as e:
        print(f"处理视频 {file} 时发生错误: {e}")
        for c in video_clips:
            c.close()
        exit()

print(f"所有视频将被截断到最短时长: {min_duration:.2f} 秒")

# 将所有剪辑的长度统一为最短时长
processed_clips = [clip.with_duration(min_duration) for clip in video_clips]

print("正在组合视频...")
grid_clips = [
    processed_clips[i:i + cols] for i in range(0, len(processed_clips), cols)
]

# 使用 clips_array 组合视频
final_clip = None
try:
    final_clip = clips_array(grid_clips)

    print(f"正在写入输出文件: {output_video_file}")
    final_clip.write_videofile(
        output_video_file,
        fps=output_fps,
        codec=codec,
        threads=4, # 可以尝试增加线程数以加快处理速度
        preset='medium' # 'ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow'
    )
    print("视频拼接完成！")

except Exception as e:
     print(f"组合或写入视频时发生错误: {e}")

finally:
    # 释放所有视频剪辑资源
    print("关闭视频资源...")
    for clip in video_clips:
        clip.close()
    # processed_clips 中的对象与 video_clips 中的是不同的（经过 set_duration 后）
    # 但 moviepy 内部可能共享资源，为保险起见，可以都关掉，但关 video_clips 即可
    if final_clip is not None:
        final_clip.close()
    print("完成。")