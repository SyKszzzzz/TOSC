
import sys

# # try:
# #     # 如果 __file__ 可用，使用脚本路径
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# # except NameError:
# #     # 如果 __file__ 不可用，使用当前工作目录
# #     BASE_DIR = os.getcwd()

# print(" 123123123123")
# print(f"BASE_DIR is {BASE_DIR}")
# # 添加两级父目录到 sys.path
# PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../"))
sys.path.append('..')
sys.path.append('/home/syks/Partial-Point-cloud-TOG-main/datasets')

# 测试导入
from datasets.misc import collate_fn_squeeze_pcd_batch_grasp
