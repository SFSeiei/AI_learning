# from collections import OrderedDict
#
# items = (('a', 1), ('b', 2), ('c', 3), ('d', 4))
# dict = dict(items)
# o_dict = OrderedDict(items)
# # dict['d'] = 12
# # dict['a'] = 21
# # dict['c'] = 24
# # dict['b'] = 36
# # o_dict['d'] = 121
# # o_dict['a'] = 412
# # o_dict['c'] = 215
# # o_dict['b'] = 344
# print(dict, o_dict)
# for k, v in dict.items():
#     print('k:', k, 'v:', v)
#
# for k, v in o_dict.items():
#     print('k:', k, 'v:', v)
#
#

# __________________test_random_________________________

import random

random.seed(45)
print("random1", random.random())
random.seed(45)
print("random2", random.random())
random.seed(45)
print("random3", random.random())
