import os
# 清除代理设置，避免本地代理未运行导致连接超时
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''
os.environ['NO_PROXY'] = '*'
# 使用国内 HuggingFace 镜像加速模型下载
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import math
import torch
import pandas as pd

# 加载数据：从CSV文件中读取英文-中文配对
def load_data(filepath):
    df = pd.read_csv(filepath, sep='\t', names=['en', 'zh'])
    return list(zip(df['en'], df['zh']))

# 示例数据（内联，用于演示；实际可用 load_data 从文件加载）
data_pairs = [
    ("The Herschel Space Observatory observed distant galaxies in infrared.", "赫歇尔空间望远镜在红外线中观测遥远的星系。"),
    ("Lunar regolith is a complex mixture of fine particles.", "月壤是由细小颗粒组成的复杂混合物。"),
    ("Infrared spectroscopy helps study the composition of celestial objects.", "红外光谱学有助于研究天体的成分。"),
    ("The satellite completed orbital insertion successfully.", "卫星成功完成了轨道插入。"),
    ("Mars has a thin atmosphere composed mostly of carbon dioxide.", "火星有一层主要由二氧化碳组成的稀薄大气层。"),
]

# 移除空值与英文文本过长的数据对
data_pairs = [(en, zh) for en, zh in data_pairs 
    if isinstance(en, str) and isinstance(zh, str) and len(en) < 1000]

from transformers import AutoTokenizer

# 使用 HuggingFace transformers 加载英文和中文的分词器
tokenizer_en = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenizer_zh = AutoTokenizer.from_pretrained("bert-base-chinese")

from collections import Counter

# 用于从数据中构建词表的辅助函数
def build_vocab(data_pairs, tokenizer, lang='src', 
    specials=["<unk>", "<pad>", "<bos>", "<eos>"]):
    counter = Counter()
    for src, tgt in data_pairs:
        text = src if lang == 'src' else tgt
        tokens = tokenizer.tokenize(text)
        counter.update(tokens)
    # 构建 token→index 映射，特殊符号在前
    vocab = {tok: i for i, tok in enumerate(specials)}
    for tok, _ in counter.most_common():
        if tok not in vocab:
            vocab[tok] = len(vocab)
    return vocab

# 构建英文词表
vocab_en = build_vocab(data_pairs, tokenizer_en, lang='src')
# 构建中文词表
vocab_zh = build_vocab(data_pairs, tokenizer_zh, lang='tgt')

# 辅助函数：将 token 列表转为 index 列表（未登录词映射为 <unk>）
def tokens_to_indices(tokens, vocab):
    unk_idx = vocab.get('<unk>', 0)
    return [vocab.get(tok, unk_idx) for tok in tokens]

# 构建中文词表的反向映射（index→token），用于解码输出
vocab_zh_rev = {i: tok for tok, i in vocab_zh.items()}


from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

# 数据预处理：将原始文本对转为张量格式，添加 <bos>/<eos>
def data_process(data):
    processed = []
    for src, tgt in data:
        src_tensor = torch.tensor([vocab_en['<bos>']] + 
            tokens_to_indices(tokenizer_en.tokenize(src), vocab_en) + [vocab_en['<eos>']])
        tgt_tensor = torch.tensor([vocab_zh['<bos>']] + 
            tokens_to_indices(tokenizer_zh.tokenize(tgt), vocab_zh) + [vocab_zh['<eos>']])
        processed.append((src_tensor, tgt_tensor))
    return processed

# 将数据处理为张量对
train_data = data_process(data_pairs)

# 自定义 collate 函数用于 DataLoader 中的动态 padding
def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, padding_value=vocab_en['<pad>'])
    tgt_batch = pad_sequence(tgt_batch, padding_value=vocab_zh['<pad>'])
    return src_batch, tgt_batch

# 构建训练数据的 DataLoader
train_iter = DataLoader(train_data, batch_size=32, shuffle=True, 
    collate_fn=collate_fn)

import torch.nn as nn
from torch.nn import Transformer

# 构建基于 Transformer 的 Seq2Seq 模型
class Seq2SeqTransformer(nn.Module):
    def __init__(self, num_layers, emb_size, nhead, src_vocab_size, 
        tgt_vocab_size, dim_feedforward=512):
        super().__init__()

        # 词嵌入层
        self.src_tok_emb = nn.Embedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size)

        # Transformer 模块（包含编码器和解码器）
        self.transformer = Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=0.1
        )

        # 输出层：将 Transformer 的输出映射到目标词表大小
        self.generator = nn.Linear(emb_size, tgt_vocab_size)

    # 前向传播函数
    def forward(self, src, tgt, src_mask, tgt_mask):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
        output = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask)
        return self.generator(output)

import math
import torch

# 位置编码模块（可加到嵌入中）
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, maxlen=5000):
        super().__init__()

        # 构建位置编码矩阵
        pe = torch.zeros(maxlen, emb_size)
        position = torch.arange(0, maxlen, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_size, 2).float() * 
            (-math.log(10000.0) / emb_size))

        pe[:, 0::2] = torch.sin(position * div_term)   # 偶数位置用 sin
        pe[:, 1::2] = torch.cos(position * div_term)   # 奇数位置用 cos

        pe = pe.unsqueeze(1)  # 添加 batch 维度
        self.register_buffer('pe', pe)  # 注册为 buffer（不训练）

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

# Encoder 层结构注释：
#    x → [多头注意力] → [Add & Norm] → [前馈层] → [Add & Norm]

# Decoder 层结构注释：
#    y → [Masked 注意力] → [Add & Norm]
#      → [交叉注意力] → [Add & Norm]
#      → [前馈层] → [Add & Norm]

# 构造下三角掩码（用于解码阶段防止看到未来）
def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
    return mask.float().masked_fill(mask == 0, 
        float('-inf')).masked_fill(mask == 1, float(0.0))

import torch.nn as nn

# 设置忽略的 PAD 索引
PAD_IDX = vocab_zh['<pad>']
# 使用交叉熵损失，忽略 PAD 部分
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

import torch
import torch.optim as optim

# 设置计算设备：优先使用 GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 实例化模型
model = Seq2SeqTransformer(
    num_layers=3, emb_size=512, nhead=8,
    src_vocab_size=len(vocab_en),
    tgt_vocab_size=len(vocab_zh)
).to(DEVICE)

# 优化器
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 再次定义掩码函数，确保在 DEVICE 上
def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    return mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

# 训练一个 epoch 的函数
def train_epoch(model, dataloader):
    model.train()
    total_loss = 0
    for src, tgt in dataloader:
        src = src.to(DEVICE)        # 源序列
        tgt = tgt.to(DEVICE)        # 目标序列

        tgt_input = tgt[:-1, :]     # 输入部分（去掉最后一个）
        tgt_output = tgt[1:, :]     # 标签部分（去掉第一个）

        # 源掩码：不屏蔽任何位置
        src_mask = torch.zeros((src.size(0), 
            src.size(0)), device=DEVICE).type(torch.bool)
        # 目标掩码：生成自回归 mask
        tgt_mask = generate_square_subsequent_mask(tgt_input.size(0)).to(DEVICE)

        # 模型前向传播
        logits = model(src, tgt_input, src_mask, tgt_mask)

        optimizer.zero_grad()
        # 调整形状：flatten 后用于交叉熵
        loss = criterion(logits.reshape(-1, logits.shape[-1])
            , tgt_output.reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# 设置训练轮数
NUM_EPOCHS = 100

# 开始训练
for epoch in range(NUM_EPOCHS):
    loss = train_epoch(model, train_iter)
    print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

# 贪婪解码函数：用于推理阶段的翻译
def greedy_decode(model, src_sentence, max_len=200):
    model.eval()
    # 编码源句
    src = torch.tensor([vocab_en['<bos>']]+tokens_to_indices(tokenizer_en.tokenize(src_sentence), vocab_en) 
    + [vocab_en['<eos>']]).unsqueeze(1).to(DEVICE)

    src_mask = torch.zeros((src.size(0), src.size(0)), 
        device=DEVICE).type(torch.bool)
    # 编码器输出
    memory = model.transformer.encoder(
        model.positional_encoding(model.src_tok_emb(src)), src_mask)
    
    # 初始化目标序列
    ys = torch.tensor([[vocab_zh['<bos>']]], device=DEVICE)

    # 逐步生成目标序列
    for i in range(max_len):
        tgt_mask = generate_square_subsequent_mask(ys.size(0)).to(DEVICE)
        out = model.transformer.decoder(
            model.positional_encoding(model.tgt_tok_emb(ys)),
            memory, tgt_mask)
        logits = model.generator(out[-1])  # 最后一个时间步预测下一个词
        next_token = torch.argmax(logits, dim=1).item()
        ys = torch.cat([ys, torch.tensor([[next_token]], device=DEVICE)], dim=0)
        if next_token == vocab_zh['<eos>']:
            break

    # 去掉 <bos> 和 <eos> 后转换成字符串
    return "".join([vocab_zh_rev.get(tok.item(), '') for tok in ys.squeeze()[1:-1]])

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

# ========== BLEU 计算公式 ==========
# BLEU (Bilingual Evaluation Understudy) 公式：
#
#   BP = 1                    若 c > r
#      = exp(1 - r/c)        若 c <= r
#
#   其中 c = 候选翻译长度, r = 参考翻译有效长度
#
#   BLEU = BP × exp( Σ_{n=1}^{N} w_n × log(p_n) )
#
#   其中 p_n 为修正的 n-gram 精确率, w_n 为权重（通常 w_n = 1/N, N=4）
# ===================================

# 计算 BLEU 分数
def compute_bleu(model, data_pairs, num_samples=100):
    candidates = []
    references = []
    for i in range(min(num_samples, len(data_pairs))):
        src, tgt = data_pairs[i]
        pred = greedy_decode(model, src)
        candidates.append(list(pred))        # 模型翻译结果（按字符拆分）
        references.append([list(tgt)])       # 参考译文（按字符拆分）
    smoothie = SmoothingFunction().method4
    return corpus_bleu(references, candidates, smoothing_function=smoothie)

# 示例天文术语词典（中英对照）
astro_terms = {
    "Herschel Space Observatory": "赫歇尔空间望远镜",
    "infrared spectroscopy": "红外光谱",
    "orbital insertion": "轨道插入",
    "lunar regolith": "月壤"
}

# 根据术语词典进行术语替换
def apply_astro_dictionary(text, term_dict):
    for en_term, zh_term in term_dict.items():
        if en_term in text:
            text = text.replace(en_term, zh_term)
    return text

# 示例：使用术语替换后的推理结果
src_sentence = "The Herschel Space Observatory observed distant galaxies in infrared."
pred = greedy_decode(model, src_sentence)
pred_postprocessed = apply_astro_dictionary(pred, astro_terms)

# 文档翻译函数：整篇英文文档逐句翻译，并可用术语替换
def translate_document(input_path, output_path, model, term_dict=None):
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    translated_lines = []
    for line in lines:
        # 将一行中的多个句子拆开翻译（按句号分隔）
        sentences = [s.strip() for s in line.strip().split('.') if s.strip()]
        for sentence in sentences:
            sentence += '.'  # 还原句号
            zh = greedy_decode(model, sentence)
            if term_dict:
                zh = apply_astro_dictionary(zh, term_dict)
            translated_lines.append(zh)

    # 写入翻译结果
    with open(output_path, 'w', encoding='utf-8') as f:
        for zh_line in translated_lines:
            f.write(zh_line + '\n')

    print(f"[✓] 翻译完成：已保存至 {output_path}")

# ========== 读取英文文件并翻译为中文 ==========
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.join(script_dir, "sample_en.txt")
output_file = os.path.join(script_dir, "sample_zh_output.txt")

translate_document(input_file, output_file, model, term_dict=astro_terms)

# 计算 BLEU 分数并输出
bleu = compute_bleu(model, data_pairs, num_samples=len(data_pairs))
print(f"BLEU score: {bleu:.4f}")
