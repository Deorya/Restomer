import os

# 要处理的文件夹路径 (请修改为您的实际路径)
folder_path = '/Deraining/Datasets/train/Rain100L/input'
# 要移除的前缀
prefix_to_remove = 'rain-'

for filename in os.listdir(folder_path):
    if filename.startswith(prefix_to_remove):
        # 构建旧的完整文件路径
        old_file = os.path.join(folder_path, filename)
        # 构建新的文件名 (去掉前缀)
        new_filename = filename[len(prefix_to_remove):]
        # 构建新的完整文件路径
        new_file = os.path.join(folder_path, new_filename)
        # 执行重命名
        os.rename(old_file, new_file)
        print(f'Renamed: {filename} -> {new_filename}')

print('Done.')

'''
**如何使用这个脚本：**
1.  将上述代码保存为一个 `.py` 文件（例如 `rename_files.py`）并放在您的项目根目录。
2.  **修改 `folder_path` 和 `prefix_to_remove` 的值**以匹配您的情况。
3.  在您的虚拟环境中运行 `python rename_files.py`。
4.  对 `input` 和 `target` 两个文件夹都进行类似的操作，直到两个文件夹中的配对文件名完全一致。

---

**总结：**

请仔细检查并确保您**训练集 `input` 和 `target` 文件夹**中的**文件名**和**文件数量**完全匹配。解决这个问题后，`AssertionError` 就会消失。
'''