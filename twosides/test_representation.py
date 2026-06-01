import os 
import pandas as pd 
import numpy as np
import torch


CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                 "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                 "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                 "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                 "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                 "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                 "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                 "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}
compound_max = 100


def label_smiles(line, smi_ch_ind, MAX_SMI_LEN=100):
    X = np.zeros(MAX_SMI_LEN, dtype=np.int64())
    for i, ch in enumerate(line[:MAX_SMI_LEN]):
        X[i] = smi_ch_ind[ch]
    return X



def get_drug_to_smiles():
    preprocessed_file_path = 'data/twosides_ge_500.zip'
    drug_data = pd.read_csv(preprocessed_file_path,delimiter=',')
    drug_to_smiles = {}
    for index, row in drug_data.T.iteritems():
        drug_id1 = row['Drug1_ID']
        drug_smiles1 = row['Drug1']
        drug_id2 = row['Drug2_ID']
        drug_smiles2 = row['Drug2']

        if drug_id1 not in drug_data.keys():
            drug_to_smiles[drug_id1] = drug_smiles1
        if drug_id2 not in drug_data.keys():
            drug_to_smiles[drug_id2] = drug_smiles2
    return drug_to_smiles




def get_drug_embedding(drug_to_smiles,drug_id):
    #print('hahahah')
    smiles = drug_to_smiles[drug_id]
    drug_value = torch.from_numpy(label_smiles(smiles, CHARISOSMISET, compound_max))
    #print(drug_value.shape)
    return drug_value

# class CQAttention(torch.nn.Module): #remove masks
#     def __init__(self, block_hidden_dim):
#         super().__init__()
#         # self.dropout = dropout
#         w4C = torch.empty(block_hidden_dim, 1)
#         w4Q = torch.empty(block_hidden_dim, 1)
#         w4mlu = torch.empty(1, 1, block_hidden_dim)
#         torch.nn.init.xavier_uniform_(w4C)
#         torch.nn.init.xavier_uniform_(w4Q)
#         torch.nn.init.xavier_uniform_(w4mlu)
#         self.w4C = torch.nn.Parameter(w4C)
#         self.w4Q = torch.nn.Parameter(w4Q)
#         self.w4mlu = torch.nn.Parameter(w4mlu)
#
#         bias = torch.empty(1)
#         torch.nn.init.constant_(bias, 0)
#         self.bias = torch.nn.Parameter(bias)
#
#     def forward(self, C, Q):
#         S = self.trilinear_for_attention(C, Q)
#         return S
#
#     def trilinear_for_attention(self, C, Q):
#         max_q_len = Q.size(-2)
#         max_context_len = C.size(-2)
#         subres0 = torch.matmul(C, self.w4C).expand([-1, -1, max_q_len])
#         subres1 = torch.matmul(Q, self.w4Q).transpose(1, 2).expand([-1, max_context_len, -1])
#         subres2 = torch.matmul(C * self.w4mlu, Q.transpose(1, 2))
#         res = subres0 + subres1 + subres2
#         res += self.bias
#         return res

if __name__ == "__main__":
    smiles = "CID000003964"
    drug_to_smiles = get_drug_to_smiles()
    print(get_drug_embedding(drug_to_smiles,smiles))
    #
    #
    # for drug_id,compoundstr in drug_to_smiles.items():
    #
    #     drug_value = torch.from_numpy(label_smiles(compoundstr, CHARISOSMISET, compound_max))
    #     print(drug_value.shape)

