import os
import shutil
import glob

source_folder = "/home/myDiffuSeq/generation_outputs/diffuseq_qqp_h128_lr0.0001_t2000_sqrt_lossaware_seed102_none20240516-18:31:48"
destination_folder = "/home/myDiffuSeq/generation_outputs/???"

# 获取源文件夹中包含'step10'的所有文件
files_to_move = glob.glob(os.path.join(source_folder, "**/*20*.json"), recursive=True)

# 遍历文件并移动到目标文件夹，保持原始文件夹结构
for file in files_to_move:
    print('move file', file)
    relative_path = os.path.relpath(file, source_folder)
    destination_path = os.path.join(destination_folder, relative_path)
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)  # 确保目标文件夹存在
    shutil.move(file, destination_path)