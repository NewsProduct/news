# 设置输入和输出文件路径
test_dataset_path = './datasets/test_dataset.csv'
p1_output_path = './datasets/test_dataset_p1.csv'
p2_output_path = './datasets/test_dataset_p2.csv'
p3_output_path = './datasets/test_dataset_p3.csv'
p4_output_path = './datasets/test_dataset_p4.csv'

# 读取测试集数据
with open(test_dataset_path, 'r', encoding='utf-8') as f:
    test_data = f.readlines()

# 计算每个子集的数据行数
total_rows = len(test_data)
rows_per_subset = total_rows // 4
remaining_rows = total_rows % 4

# 将数据划分为4个子集
p1_data = test_data[:rows_per_subset]
p2_data = test_data[rows_per_subset:rows_per_subset*2]
p3_data = test_data[rows_per_subset*2:rows_per_subset*3]
p4_data = test_data[rows_per_subset*3:]

# 如果有剩余数据,则添加到最后一个子集
if remaining_rows > 0:
    p4_data.extend(test_data[-remaining_rows:])

# 保存子集数据到文件
with open(p1_output_path, 'w', encoding='utf-8') as f:
    f.writelines(p1_data)

with open(p2_output_path, 'w', encoding='utf-8') as f:
    f.writelines(p2_data)

with open(p3_output_path, 'w', encoding='utf-8') as f:
    f.writelines(p3_data)

with open(p4_output_path, 'w', encoding='utf-8') as f:
    f.writelines(p4_data)
