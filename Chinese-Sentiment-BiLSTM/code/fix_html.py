"""
删除 advanced_app.html 中重复的"模型架构"和"数据统计"内容块
"""

with open('advanced_app.html', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 找到第二个"模型架构"块的位置（在 1400 行之后）
start_idx = None
for i in range(1400, len(lines)):
    if '<!-- 模型架构 -->' in lines[i]:
        start_idx = i
        break

if start_idx:
    # 找到"<!-- 页脚 -->"的位置
    end_idx = None
    for i in range(start_idx, len(lines)):
        if '<!-- 页脚 -->' in lines[i]:
            end_idx = i
            break
    
    if end_idx:
        # 删除从 start_idx 到 end_idx-1 的内容（保留页脚）
        new_lines = lines[:start_idx] + lines[end_idx:]
        with open('advanced_app.html', 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        print(f'成功删除从第{start_idx+1}行到第{end_idx}行的重复内容')
    else:
        print('未找到页脚标记')
else:
    print('未找到第二个模型架构标记，文件可能已经处理过')
