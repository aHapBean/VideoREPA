from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.compositing.CompositeVideoClip import clips_array
import os

# --- 配置 ---
# 假设你的12个视频文件在这里
# 请根据你的实际文件名修改列表
input_video_files = [
    "mn_hunyuan_1.mp4", "mn_cogvideox_1.mp4", "mn_videorepa_1.mp4",
    "mn_hunyuan_2.mp4", "mn_cogvideox_2.mp4", "mn_videorepa_2.mp4",
    "mn_hunyuan_3.mp4", "mn_cogvideox_3.mp4", "mn_videorepa_3.mp4",
    "mn_hunyuan_4.mp4", "mn_cogvideox_4.mp4", "mn_videorepa_4.mp4"
]

output_video_file = "output_grid_video.mp4"

# 网格布局 (4行, 3列)
rows = 4
cols = 3

# 可选：如果你的视频尺寸不同，可以设置一个目标尺寸让它们统一
# 例如： target_size = (640, 480)
# 如果所有视频尺寸相同，可以设置为 None，代码会自动获取第一个视频的尺寸
target_size = None # 例如：(640, 480) 或 None

# 可选：设置输出视频的帧率 (FPS)
# 如果设置为 None，会尝试使用输入视频的帧率，但有时设置一个明确的值更好
output_fps = 24 # 或 None

# 可选：设置输出视频的编码器 (推荐使用 libx264 或 libvpx)
# codec = "libx264" # 常用于 MP4
codec = "libvpx"   # 常用于 WebM，兼容性更好

# --- 脚本开始 ---

if len(input_video_files) != rows * cols:
    print(f"错误: 输入文件数量 ({len(input_video_files)}) 与设定的网格 ({rows}x{cols}={rows*cols}) 不匹配。")
    exit()

# 确保所有输入文件存在
for file in input_video_files:
    if not os.path.exists(file):
        print(f"错误: 文件 {file} 不存在。")
        exit()

print("正在加载视频...")
video_clips = []
target_size = [720, 480]
for i, file in enumerate(input_video_files):
    print(f"加载并处理视频: {file} ({i+1}/{len(input_video_files)})")
    try:
        clip = VideoFileClip(file)

        # 获取第一个视频的尺寸作为基准（如果 target_size 未设置）
        if target_size is None and i == 0:
             target_size = clip.size
             print(f"使用第一个视频的尺寸作为目标尺寸: {target_size}")
        elif target_size is None:
            # 如果是后续视频且 target_size 为 None，我们期望它们的尺寸和第一个视频一样
            if clip.size != target_size:
                 print(f"警告: 视频 {file} 的尺寸 {clip.size} 与第一个视频的尺寸 {target_size} 不同。这可能导致布局问题。")
                 # 可以选择在这里停止或强制缩放，这里选择继续，但尺寸不一致可能导致问题
                 # 如果要强制缩放，可以将上面 target_size = None 改为 target_size = clip.size. For now, let's assume they are the same size or handle resizing below.

        # 如果设置了 target_size，则进行缩放
        if target_size is not None and clip.size != target_size:
             # 注意: 简单的 resize 会改变宽高比，如果想保持比例并填充黑边，需要更复杂的处理
             # 这里为了简单，直接缩放到目标尺寸，可能会导致拉伸或压缩
             print(f"缩放视频 {file} 从 {clip.size} 到 {target_size}")
             clip.resized(new_size=target_size)


        video_clips.append(clip)
    except Exception as e:
        print(f"处理视频 {file} 时发生错误: {e}")
        # 关闭已加载的视频资源以避免内存泄漏
        for c in video_clips:
            c.close()
        exit()


# 确保所有视频长度相同 (用户已承诺)
# 可以加一个检查，但不强制要求，moviepy 在 clips_array 中依赖于相同长度
duration = video_clips[0].duration
for i, clip in enumerate(video_clips):
    if abs(clip.duration - duration) > 0.1: # 允许一点点浮动误差
        print(f"警告: 视频 {input_video_files[i]} 的时长 ({clip.duration}) 与第一个视频 ({duration}) 不同。")
        print("这可能会导致问题，因为 clips_array 要求所有视频时长相同。")
        # 可以选择在这里截断或填充，这里选择继续，依赖用户确保长度一致

print("正在组合视频...")

# 将一维列表的视频剪辑组织成二维列表 (网格)
# 例如: [[clip0, clip1, clip2], [clip3, clip4, clip5], ...]
grid_clips = []
for r in range(rows):
    row_clips = []
    for c in range(cols):
        index = r * cols + c
        row_clips.append(video_clips[index])
    grid_clips.append(row_clips)

# 使用 clips_array 组合视频
try:
    final_clip = clips_array(grid_clips)

    # 写入最终的视频文件
    print(f"正在写入输出文件: {output_video_file}")
    final_clip.write_videofile(
        output_video_file,
        fps=output_fps if output_fps is not None else final_clip.fps, # 使用设定的FPS或合成后的FPS
        codec=codec
    )

    print("视频拼接完成！")

except Exception as e:
     print(f"组合或写入视频时发生错误: {e}")

finally:
    # 释放所有视频剪辑资源
    print("关闭视频资源...")
    for clip in video_clips:
        clip.close()
    if 'final_clip' in locals() and final_clip is not None:
        final_clip.close()
    print("完成。")