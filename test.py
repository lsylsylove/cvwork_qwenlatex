import os
import pandas as pd

# 定义文件夹和文件路径
image_dir_train = '/root/autodl-tmp/project/nougat-latex-ocr/open-dataset/test'
label_file = '/root/autodl-tmp/project/nougat-latex-ocr/open-dataset/math.txt'
output_csv = '/root/autodl-tmp/project/nougat-latex-ocr/latex_ocr_test.csv'

# 读取标签文件
with open(label_file, 'r', encoding='utf-8') as f:
    labels = f.readlines()

# 获取所有训练集图片的文件名，按数字排序
image_files_train = sorted(os.listdir(image_dir_train), key=lambda x: int(x.split('.')[0]))

# 输出图片和标签数量进行调试
print(f'训练集图片数量: {len(image_files_train)}')
print(f'标签文件行数: {len(labels)}')

# 初始化列表来存储图片路径和标签
image_paths = []
texts = []

# 遍历训练集图片文件和标签，选择train标签
for image_file in image_files_train:
    # 从图片文件名中提取后6位数字
    image_number_str = image_file[-10:-4]  # 获取文件名的最后6位数字
    image_number = int(image_number_str)  # 转换为整数
    
    # 对应的标签行是 image_number + 1（因为math.txt中的行是1-based索引，Python索引是0-based）
    label_index = image_number  # 获取math.txt中的标签行号（1-based），这里Python索引为0-based
    
    # 确保标签行存在
    if label_index < len(labels):
        label_line = labels[label_index]
        
        # 去除标签中的换行符和多余的空格
        label_text = label_line.strip()  # 去除前后空格和换行符
        
        # 去除文本中的双引号
        label_text = label_text.replace('"', '')  # 去除双引号

        # 获取图片路径
        image_path = os.path.join(image_dir_train, image_file)
        
        # 检查图片路径是否存在
        if os.path.exists(image_path):
            image_paths.append(image_path)
            texts.append(label_text)
        else:
            print(f"图片文件不存在: {image_path}")
        
        # 每处理50个样本输出进度
        if len(image_paths) % 50 == 0:
            print(f'Processed {len(image_paths)}/{len(image_files_train)} images')

# 检查是否有有效数据
if len(image_paths) == 0:
    print("没有有效的图片和标签匹配，请检查路径和标签文件格式。")
else:
    # 创建DataFrame并保存为CSV文件
    df = pd.DataFrame({
        'image_path': image_paths,
        'text': texts,
    })

    # 保存为CSV文件，使用引号包裹字段并正确处理特殊字符
    df.to_csv(output_csv, index=False, quoting=1, escapechar='\\')  # quoting=1 对所有字段加引号

    print(f'数据处理完成，已保存到 {output_csv}')
