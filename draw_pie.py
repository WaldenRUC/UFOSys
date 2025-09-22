# python draw_pie.py --input_fn result.json --dataset Wikibio --source ChatGPT --scenario web knowledge
import matplotlib.pyplot as plt
import seaborn as sns
import argparse, json
parser = argparse.ArgumentParser()
parser.add_argument('--input_fn')
parser.add_argument('--dataset')
parser.add_argument('--source')
parser.add_argument('--scenario', nargs='+')
args = parser.parse_args()

# 设置Seaborn风格（美化背景和字体）
sns.set_style("whitegrid")
sns.set_palette("pastel")

TITLE_FONTSIZE = 30
LABEL_FONTSIZE = 22  # 标签文字大小

counts = [0, 0, 0, 0, 0, 0]
# counts = [712, 446, 356, 198, 512, 340]  # 示例：Wikibio - ChatGPT
labels = ['Human (T)', 'Human (F)', 'Web (T)', 'Web (F)', 'Knowledge (T)', 'Knowledge (F)']
def build_label(source_flag, flag):
	source = source_flag.capitalize()   # 首字母大写，和你的列表一致
	tf = "T" if flag == 1 else "F"
	return f"{source} ({tf})"

with open(args.input_fn, 'r', encoding='utf-8') as fp:
	data = json.load(fp)
for line in data:
    _outputs = line['output']
    for _output in _outputs:
        source_flag = None
        flag = 0
        detail_keys = [f'{_scenario}_details' for _scenario in args.scenario]
        extraction_keys = [f'{_scenario}_extractions' for _scenario in args.scenario]
        for extraction_key, detail_key in zip(extraction_keys, detail_keys):
            # web_extractions, web_details
            source_flag = extraction_key.split('_')[0]
            extractions, details = _output[extraction_key], _output[detail_key]
            valid_judgments = [bool(detail['factuality']) for extraction, detail in zip(extractions, details) if 'noans' not in extraction['answer'].lower()]
            if len(valid_judgments) == 0: continue
            else:
                # majority voting
                flag = 1 if sum(valid_judgments) > len(valid_judgments) / 2 else 0
                break
        label_index = labels.index(build_label(source_flag, flag))
        counts[label_index] += 1

# 删除无用的counts与labels
del_indices = [_id for _id, c in enumerate(counts) if c == 0]
counts = [c for _id, c in enumerate(counts) if _id not in del_indices]
labels = [l for _id, l in enumerate(labels) if _id not in del_indices]




# 创建画布
fig, ax = plt.subplots(figsize=(8, 8))

# 绘制饼图
ax.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90,
       wedgeprops={'linewidth': 1, 'edgecolor': 'white'},
       textprops={'fontsize': LABEL_FONTSIZE})

# 设置标题
ax.set_title(f'{args.dataset} ({args.source})', fontsize=TITLE_FONTSIZE, pad=20)

# 保存图片
plt.savefig("pie_chart.pdf", dpi=300, bbox_inches='tight')

# 显示图形
plt.show()