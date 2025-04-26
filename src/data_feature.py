# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 14:28:17 2022

@author: lenovo
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 02:19:52 2021

@author: joy
"""



def AAI_embedding(seq, max_len=50):
    # 读取AAindex.txt文件
    f = open('data/AAindex.txt')
    text = f.read()
    f.close()

    # 处理文件内容
    text = text.split('\n')
    while '' in text:
        text.remove('')
    cha = text[0].split('\t')
    while '' in cha:
        cha.remove('')
    cha = cha[1:]
    index = []
    for i in range(1, len(text)):
        temp = text[i].split('\t')
        while '' in temp:
            temp.remove('')
        temp = temp[1:]
        for j in range(len(temp)):
            temp[j] = float(temp[j])
        index.append(temp)
    index = np.array(index)

    # 创建AAI字典
    AAI_dict = {}
    for j in range(len(cha)):
        AAI_dict[cha[j]] = index[:, j]

    # 添加默认的全零向量用于未知氨基酸字母
    AAI_dict['X'] = np.zeros(531)

    all_embeddings = []
    for each_seq in seq:
        temp_embeddings = []
        for each_char in each_seq:
            # 检查每个字符是否在AAI_dict中，如果不在则使用全零向量
            temp_embeddings.append(AAI_dict.get(each_char, np.zeros(531)))
        if max_len > len(each_seq):
            zero_padding = np.zeros((max_len - len(each_seq), 531))
            data_pad = np.vstack((temp_embeddings, zero_padding))
        elif max_len == len(each_seq):
            data_pad = temp_embeddings
        else:
            data_pad = temp_embeddings[:max_len]
        all_embeddings.append(data_pad)

    all_embeddings = np.array(all_embeddings)
    return torch.from_numpy(all_embeddings).float()


def PAAC_embedding(seq, max_len=50):
    # 读取PAAC.txt文件
    with open('data/PAAC.txt') as f:
        text = f.read()

    # 处理文件内容
    text = text.split('\n')
    while '' in text:
        text.remove('')
    cha = text[0].split('\t')
    while '' in cha:
        cha.remove('')
    cha = cha[1:]
    index = []
    for i in range(1, len(text)):
        temp = text[i].split('\t')
        while '' in temp:
            temp.remove('')
        temp = temp[1:]
        for j in range(len(temp)):
            temp[j] = float(temp[j])
        index.append(temp)
    index = np.array(index)

    # 创建PAAC字典
    PAAC_dict = {}
    for j in range(len(cha)):
        PAAC_dict[cha[j]] = index[:, j]

    # 添加默认的全零向量用于未知氨基酸字母
    PAAC_dict['X'] = np.zeros(3)

    all_embeddings = []
    for each_seq in seq:
        temp_embeddings = []
        for each_char in each_seq:
            # 检查每个字符是否在PAAC_dict中，如果不在则使用全零向量
            temp_embeddings.append(PAAC_dict.get(each_char, np.zeros(3)))
        if max_len > len(each_seq):
            zero_padding = np.zeros((max_len - len(each_seq), 3))
            data_pad = np.vstack((temp_embeddings, zero_padding))
        elif max_len == len(each_seq):
            data_pad = temp_embeddings
        else:
            data_pad = temp_embeddings[:max_len]
        all_embeddings.append(data_pad)

    all_embeddings = np.array(all_embeddings)
    return torch.from_numpy(all_embeddings).float()


def PC6_embedding(seq, max_len=50):
    # 读取6-pc文件
    with open('data/6-pc') as f:
        text = f.read()

    # 处理文件内容
    text = text.split('\n')
    while '' in text:
        text.remove('')
    text = text[1:]

    # 创建PC6字典
    PC6_dict = {}
    for each_line in text:
        temp = each_line.split(' ')
        while '' in temp:
            temp.remove('')
        for i in range(1, len(temp)):
            temp[i] = float(temp[i])
        PC6_dict[temp[0]] = temp[1:]

    # 添加默认的全零向量用于未知氨基酸字母
    PC6_dict['X'] = np.zeros(6)

    all_embeddings = []
    for each_seq in seq:
        temp_embeddings = []
        for each_char in each_seq:
            # 检查每个字符是否在PC6_dict中，如果不在则使用全零向量
            temp_embeddings.append(PC6_dict.get(each_char, np.zeros(6)))
        if max_len > len(each_seq):
            zero_padding = np.zeros((max_len - len(each_seq), 6))
            data_pad = np.vstack((temp_embeddings, zero_padding))
        elif max_len == len(each_seq):
            data_pad = temp_embeddings
        else:
            data_pad = temp_embeddings[:max_len]
        all_embeddings.append(data_pad)

    all_embeddings = np.array(all_embeddings)
    return torch.from_numpy(all_embeddings).float()


def BLOSUM62_embedding(seq, max_len=50):
    # 读取blosum62.txt文件
    with open('data/blosum62.txt') as f:
        text = f.read()

    # 处理文件内容
    text = text.split('\n')
    while '' in text:
        text.remove('')

    # 获取氨基酸字符
    cha = text[0].split(' ')
    while '' in cha:
        cha.remove('')

    # 处理矩阵数据
    index = []
    for i in range(1, len(text)):
        temp = text[i].split(' ')
        while '' in temp:
            temp.remove('')
        for j in range(len(temp)):
            temp[j] = float(temp[j])
        index.append(temp)
    index = np.array(index)

    # 创建BLOSUM62字典
    BLOSUM62_dict = {}
    for j in range(len(cha)):
        BLOSUM62_dict[cha[j]] = index[:, j]

    # 添加默认的全零向量用于未知氨基酸字母
    BLOSUM62_dict['X'] = np.zeros(23)

    all_embeddings = []
    for each_seq in seq:
        temp_embeddings = []
        for each_char in each_seq:
            # 检查每个字符是否在BLOSUM62_dict中，如果不在则使用全零向量
            temp_embeddings.append(BLOSUM62_dict.get(each_char, np.zeros(23)))
        if max_len > len(each_seq):
            zero_padding = np.zeros((max_len - len(each_seq), 23))
            data_pad = np.vstack((temp_embeddings, zero_padding))
        elif max_len == len(each_seq):
            data_pad = temp_embeddings
        else:
            data_pad = temp_embeddings[:max_len]
        all_embeddings.append(data_pad)

    all_embeddings = np.array(all_embeddings)
    return torch.from_numpy(all_embeddings).float()


def AAC_embedding(sequences, order='ACDEFGHIKLMNPQRSTVWY', max_len=50):
    # Define the amino acids order if it's not provided
    AA = order

    # Prepare to collect the AAC encodings
    aac_encodings = []

    for seq in sequences:
        # Remove potential '-' characters for alignment gaps
        sequence = re.sub('-', '', seq)

        # Ensure each sequence is exactly max_len characters
        if len(sequence) > max_len:
            sequence = sequence[:max_len]
        elif len(sequence) < max_len:
            sequence += 'X' * (max_len - len(sequence))  # Pad with 'X' if not enough length

        # Initialize frequency list for this sequence
        temp_encoding = np.zeros((max_len, len(AA)))

        # Calculate frequency for each position up to max_len
        for idx, char in enumerate(sequence):
            if char in AA:
                aa_index = AA.index(char)
                temp_encoding[idx, aa_index] = 1  # Set 1 for the existing amino acid position
            else:
                temp_encoding[idx, :] = 0  # Set all zeros for unknown amino acid position

        aac_encodings.append(temp_encoding)

    # Convert list of encodings to a numpy array
    all_encodings = np.array(aac_encodings)

    return torch.tensor(all_encodings, dtype=torch.float)





