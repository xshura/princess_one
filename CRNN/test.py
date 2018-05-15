import tensorflow as tf

# print(help(tf.global_variables()))
"""获取程序中所有的变量，返回的是一个list"""
# print(help(tf.squeeze))
"""
    用来张量降维
    For example:
    
    ```python
    # 't' is a tensor of shape [1, 2, 1, 3, 1, 1]
    tf.shape(tf.squeeze(t))  # [2, 3]
    ```
"""
# print(help(tf.edit_distance))
"""
    为稀疏矩阵计算，规范编辑距离
    This operation takes variable-length sequences (`hypothesis` and `truth`),
    each provided as a `SparseTensor`, and computes the Levenshtein distance.
    You can normalize the edit distance by length of `truth` by setting
    `normalize` to true.
    
"""
# print(help(tf.nn.ctc_beam_search_decoder))
"""
    ctc贪婪解码是一种特例
    **Note** The `ctc_greedy_decoder` is a special case of the
    `ctc_beam_search_decoder` with `top_paths=1` and `beam_width=1` (but
    that decoder is faster for this special case).
    * `A B` if `merge_repeated = True`.
    * `A B B B B` if `merge_repeated = False`.
"""
print(help(tf.sparse_tensor_to_dense))
"""
    将稀疏矩阵转换为张量
    Converts a `SparseTensor` into a dense tensor.
    For example, if `sp_input` has shape `[3, 5]` and non-empty string values:
    
        [0, 1]: a
        [0, 3]: b
        [2, 0]: c
    
    and `default_value` is `x`, then the output will be a dense `[3, 5]`
    string tensor with values:
    
        [[x a x b x]
         [x x x x x]
         [c x x x x]]
"""
