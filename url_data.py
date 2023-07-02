import pandas as pd

res_file1 = r"data/urlDataset.csv"
res_file2 = r"data/newUrls.csv"

df1 = pd.read_csv(res_file1)

df = df1[df1['label']==0].sample(20000)
dff = df1[df1['label']!=0]

res_df = pd.concat([df, dff]).reset_index(drop=True)
print(res_df.label.value_counts())
print(res_df.url.value_counts())
res_df.drop_duplicates(subset='url',inplace=True)
print(res_df.label.value_counts())
print(res_df.url.value_counts())

res_df.to_csv(res_file2, index=False)

# df['url'] = df['url'].map(str.strip)
# df.to_csv(res_file3, index=False)

# df['len'] = df['url'].map(len)
# df2 = df[df['len']<=50]
# print(df2.label.value_counts())
# df2.drop('len', axis=1, inplace=True) #改变原始数据
# df2.to_csv('data/urlDatasets.csv', index=False)
# df1 = pd.read_csv(res_file1)
# df2 = pd.read_csv(res_file2)
#
# df1 = df1[df1['label']==0]
# df = df1.sample(30000).reset_index(drop=True)
# res_df = pd.concat([df, df2]).reset_index(drop=True)
# res_df.to_csv(res_file3, index=False)

exit(0)

urls_file1 = r"C:\Users\14389\Desktop\urls dataset\train1.csv"
urls_file2 = r"C:\Users\14389\Desktop\urls dataset\train2.csv"
urls_file3 = r"C:\Users\14389\Desktop\urls dataset\train3.xlsx"

save_file = r"C:\Users\14389\Desktop\urls dataset\urls.csv"

labels = ['正常', '购物消费', '婚恋交友', '假冒身份', '钓鱼网站', '冒充公检法', '平台诈骗', '招聘兼职', '杀猪盘', '博彩赌博', '信贷理财', '刷单诈骗', '中奖诈骗']
labels_dict = {lb: labels.index(lb) for lb in labels}

df1 = pd.read_csv(urls_file1)
df2 = pd.read_csv(urls_file2)
df3 = pd.read_excel(urls_file3)

df1.dropna(axis="index", how='any', inplace=True) #删除空行
df2.drop('文本', axis=1, inplace=True) #改变原始数据
df2.dropna(axis="index", how='any', inplace=True) #删除空行
df3.dropna(axis="index", how='any', inplace=True) #删除空行

df1.columns = ['url', 'label']
df1['label'] = df1['label'].astype('int64')

df2.columns = ['url', 'label']
df2['label'] = df2['label'].map(labels_dict)
df2.dropna(axis="index", how='any', inplace=True) #删除空行
df2['label'] = df2['label'].astype('int64')

df3.columns = ['url', 'text', 'label']
df3.drop('text', axis=1, inplace=True) #改变原始数据

res_df = pd.concat([df1, df2, df3]).reset_index(drop=True)
res_df.to_csv(save_file, index=False)
