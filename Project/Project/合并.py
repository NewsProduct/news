p1_output_path = './submission_p1.csv'
p2_output_path = './submission_p2.csv'
p3_output_path = './submission_p3.csv'
p4_output_path = './submission_p4.csv'

# 读取四个子集文件
with open(p1_output_path, 'r', encoding='utf-8') as f:
    p1_data = f.readlines()

with open(p2_output_path, 'r', encoding='utf-8') as f:
    p2_data = f.readlines()

with open(p3_output_path, 'r', encoding='utf-8') as f:
    p3_data = f.readlines()

with open(p4_output_path, 'r', encoding='utf-8') as f:
    p4_data = f.readlines()

# 合并四个子集
merged_data = p1_data + p2_data + p3_data + p4_data

# 保存合并后的数据
with open('submission.csv', 'w', encoding='utf-8') as f:
    f.writelines(merged_data)