import matplotlib.pyplot as plt
import seaborn as sns

# 设置Seaborn风格（美化背景和字体）
sns.set_style("whitegrid")
sns.set_palette("pastel")

TITLE_FONTSIZE = 30
LABEL_FONTSIZE = 22  # 标签文字大小

# 数据（你可以换成 data2、data3、data4）
data = [712, 446, 356, 198, 512, 340]  # 示例：Wikibio - ChatGPT
labels = ['Human (T)', 'Human (F)', 'Web (T)', 'Web (F)', 'Knowledge (T)', 'Knowledge (F)']

# 创建画布
fig, ax = plt.subplots(figsize=(8, 8))

# 绘制饼图
ax.pie(data, labels=labels, autopct='%1.1f%%', startangle=90,
       wedgeprops={'linewidth': 1, 'edgecolor': 'white'},
       textprops={'fontsize': LABEL_FONTSIZE})

# 设置标题
ax.set_title('Wikibio (ChatGPT)', fontsize=TITLE_FONTSIZE, pad=20)

# 保存图片
plt.savefig("pie_chart.pdf", dpi=300, bbox_inches='tight')

# 显示图形
plt.show()