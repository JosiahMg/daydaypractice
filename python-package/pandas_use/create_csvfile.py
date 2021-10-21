"""
通过字典的方式生成csv文件

"""
import pandas as pd
import config

df = pd.DataFrame(columns=["epoch", "train_loss", "train_auc", "test_loss", "test_auc"], index=None)

log_dic = {"epoch": 2,
           "train_loss": 0.1,
           "train_auc": 1.,
           "test_loss": 23,
           "test_auc": 1
          }

# 此次需要注意df.append()不会修改df的值，返回值才是追加的内容
df = df.append([log_dic])

# 需要index=False
df.to_csv(config.dict_csv_path, sep=',', index=False)

