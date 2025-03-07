# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
import lightgbm as lgb
import xgboost as xgb
# 检查GPU是否可用
print(xgb.__version__)  # 应显示版本号（如2.0.0）
print('GPU available:', xgb.XGBClassifier().get_params()['gpu_id'] is not None)

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

