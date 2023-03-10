import torch
from transformers import BertTokenizer, AutoModel

tokenizer = BertTokenizer.from_pretrained("indobenchmark/indobert-base-p2")
bertModel = AutoModel.from_pretrained("./BERTmodel/")

x = torch.LongTensor(tokenizer.encode('aku adalah anak [MASK]')).view(1,-1)
print(x, bertModel(x)[0].sum())

