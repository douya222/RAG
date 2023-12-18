import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

'''
模型性能比较：

chatglm3+rag 在大多数指标上表现最好，而 llama2-7b 在大多数指标上表现最差。
使用rag（可能是retrieval-augmented generation）的模型似乎在大多数情况下表现更好。
其他注意事项：

baichuan2+rag 也在多个指标上表现非常好。
llama2-7b 的性能相对较低，尤其是在BLEU-4和ROUGE指标上。
'''

#            model  BLEU-4  ROUGE_1  ROUGE_2  ROUGE_L  precision  recall  f1_score
# 0    chatglm2-6b   6.441   32.646   12.821   23.229      0.727   0.652     0.685
# 1   chatglm2+rag  25.721   50.953   35.493   45.698      0.800   0.766     0.778
# 2    chatglm3-6b   7.306   32.003   13.719   22.939      0.735   0.663     0.694
# 3   chatglm3+rag  27.038   54.188   37.282   49.139      0.781   0.805     0.790
# 4   baichuan2-7b   9.617   35.498   15.847   28.405      0.739   0.673     0.703
# 5  baichuan2+rag  26.652   55.905   37.947   50.745      0.799   0.802     0.796
# 6      llama2-7b   0.340   11.193    1.214    7.120      0.614   0.538     0.569
# 7     llama2+rag   7.302   20.156   10.631   18.340      0.595   0.608     0.600

# 读取xlsx文件
# df = pd.read_excel('./data.xlsx') 

# Save the figure
plt.savefig('./res/8.png')






