import sys
item_input = sys.stdin.readlines()
num = int(item_input[0]) 
str_list = item_input[1].split()
sorted_list = sorted(str_list)
print(" ".join(sorted_list))

# 输入
# 5
# c d a bb e

# 输出
# a bb c d e

# def rebbit(n):
#     if n == 1:
#         return 2
#     elif n == 2:
#         return 3
#     else:
#         return rebbit(n-1)+rebbit(n-2)
# result  = rebbit(3)
# print(result)