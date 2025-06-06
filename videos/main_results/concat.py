from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from moviepy.video.compositing.CompositeVideoClip import clips_array
from moviepy.video.VideoClip import TextClip, ColorClip # 导入必要的类
import os

# --- 配置 ---
input_video_files = [
    "mn_hunyuan_1.mp4", "mn_cogvideox_1.mp4", "mn_videorepa_1.mp4", 
    "mn_hunyuan_2.mp4", "mn_cogvideox_2.mp4", "mn_videorepa_2.mp4",
    "mn_hunyuan_3.mp4", "mn_cogvideox_3.mp4", "mn_videorepa_3.mp4",
    "mn_hunyuan_4.mp4", "mn_cogvideox_4.mp4", "mn_videorepa_4.mp4"
]
output_video_file = "output_grid_video_with_titles.mp4"

# 网格布局
rows = 4
cols = 3

# 单个视频的目标尺寸
target_size = (720, 480)

# 输出视频的帧率 (FPS)
output_fps = 24

# 编码器
codec = "libx264"

# --- 新增：标题栏配置 ---
model_names = ["HunyuanVideo", "CogVideoX", "Ours"]
header_height = 60  # 标题栏的高度（像素）
font_size = 40      # 字体大小
font_color = 'white'
# 如果系统找不到默认字体，可以在这里指定字体文件路径，例如：font='path/to/your/font.ttf'
# 常见的字体名称: 'Arial', 'Helvetica', 'Courier'
font = 'Arial' 
bg_color = 'black'  # 标题栏背景色

# --- 脚本开始 ---

if len(input_video_files) != rows * cols:
    print(f"错误: 输入文件数量 ({len(input_video_files)}) 与设定的网格 ({rows}x{cols}={rows*cols}) 不匹配。")
    exit()
if len(model_names) != cols:
    print(f"错误: 模型名称数量 ({len(model_names)}) 与列数 ({cols}) 不匹配。")
    exit()

for file in input_video_files:
    if not os.path.exists(file):
        print(f"错误: 文件 {file} 不存在。")
        exit()

print("正在加载和处理视频...")
video_clips = []
min_duration = float('inf')

for i, file in enumerate(input_video_files):
    print(f"处理视频: {file} ({i+1}/{len(input_video_files)})")
    try:
        clip = VideoFileClip(file)

        # 【已修正】确保缩放操作生效
        if clip.size != target_size:
             print(f"缩放视频 {file} 从 {clip.size} 到 {target_size}")
             clip.resized(new_size=target_size)
        
        if clip.duration < min_duration:
            min_duration = clip.duration
            
        video_clips.append(clip)
    except Exception as e:
        print(f"处理视频 {file} 时发生错误: {e}")
        for c in video_clips: c.close()
        exit()

print(f"所有视频将被截断到最短时长: {min_duration:.2f} 秒")

# 【已修正】统一所有剪辑的长度
processed_clips = [clip.with_duration(min_duration) for clip in video_clips]

print("正在组合视频网格...")
grid_clips = [
    processed_clips[i:i + cols] for i in range(0, len(processed_clips), cols)
]

# 1. 首先，创建视频网格剪辑
video_grid_clip = clips_array(grid_clips)

# --- 新增逻辑：创建并添加标题栏 ---
print("正在创建标题栏...")

# 2. 创建文本剪辑列表
text_clips = []
for name in model_names:
    txt_clip = TextClip(
        name,
        # fontsize=font_size,
        color=font_color,
        # font=font,
        size=(target_size[0], header_height) # 让每个文本剪辑和列宽一样，方便对齐
    ).with_duration(min_duration)
    text_clips.append(txt_clip)

# 3. 将文本剪辑水平排列，形成一个完整的标题栏
#    我们不需要背景，因为TextClip默认背景是透明的，我们可以直接把它们放在一个纯色背景上
header_row = clips_array([text_clips])

# 4. 创建一个黑色的背景，并把标题栏组合上去
header_bg = ColorClip(
    size=(video_grid_clip.w, header_height), 
    color=bg_color, 
    duration=min_duration
)

header_clip = CompositeVideoClip([header_bg, header_row])


# 5. 最后，将标题栏和视频网格垂直堆叠
print("正在组合标题栏和视频网格...")
final_clip = clips_array([[header_clip], [video_grid_clip]])


# --- 写入文件和清理 ---
try:
    print(f"正在写入最终输出文件: {output_video_file}")
    final_clip.write_videofile(
        output_video_file,
        fps=output_fps,
        codec=codec,
        threads=4,
        preset='medium'
    )
    print("视频拼接完成！")

except Exception as e:
     print(f"组合或写入视频时发生错误: {e}")

finally:
    print("关闭所有视频资源...")
    # 关闭所有加载的原始剪辑
    for clip in video_clips:
        clip.close()
    # 关闭所有中间生成的剪辑
    if 'video_grid_clip' in locals(): video_grid_clip.close()
    if 'header_clip' in locals(): header_clip.close()
    if 'final_clip' in locals(): final_clip.close()
    print("完成。")