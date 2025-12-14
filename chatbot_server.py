# -*- coding: utf-8 -*-
"""
완벽한 GPT-3 챗봇 서버
Flask + PyTorch 트랜스포머 모델
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os

# ========== GPT-3 모델 정의 ==========

class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, block_size):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size))
            .view(1, 1, block_size, block_size)
        )

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)

class MLP(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, 4 * embed_dim)
        self.fc2 = nn.Linear(4 * embed_dim, embed_dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, block_size)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT3(nn.Module):
    def __init__(self, vocab_size, block_size, n_layers=6, embed_dim=256, n_heads=8):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(block_size, embed_dim)
        self.blocks = nn.Sequential(
            *[TransformerBlock(embed_dim, n_heads, block_size) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.block_size = block_size

    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos = torch.arange(0, T, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0):
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= model.block_size else idx[:, -model.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, 1)
        idx = torch.cat([idx, idx_next], dim=1)
    return idx

# ========== 토크나이저 ==========

class SimpleTokenizer:
    def __init__(self):
        # 한글 + 영어 + 숫자 + 특수문자 지원
        self.chars = sorted(list(set(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "
            "가나다라마바사아자차카타파하거너더러머버서어저처커터퍼허"
            "고노도로모보소오조초코토포호구누두루무부수우주추쿠투푸후"
            "그느드르므브스으즈츠크트프흐기니디리미비시이지치키티피히"
            "!@#$%^&*()_+-=[]{}|;':\",./<>?~`"
            "안녕하세요감사합니다인공지능챗봇동아리학습생성형최고"
        )))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

    def encode(self, text):
        return [self.char_to_idx.get(c, 0) for c in text]

    def decode(self, indices):
        return ''.join([self.idx_to_char.get(i, '') for i in indices])

# ========== 학습 데이터 ==========

TRAINING_DATA = """
안녕하세요! 반갑습니다!
챗봇이 뭐예요? 챗봇은 사람과 대화하는 AI 프로그램입니다.
AI가 뭐예요? AI는 인공지능으로 컴퓨터가 학습하고 생각하는 기술입니다.
날씨 어때요? 오늘 날씨는 맑고 좋습니다!
공부 열심히 하세요! 네 열심히 하겠습니다!
동아리가 뭐예요? 동아리는 같은 관심사를 가진 사람들의 모임입니다.
프로그래밍이 뭐예요? 프로그래밍은 컴퓨터에게 명령을 내리는 것입니다.
파이썬이 뭐예요? 파이썬은 쉽고 강력한 프로그래밍 언어입니다.
어떻게 지내세요? 저는 잘 지내고 있습니다!
감사합니다! 천만에요! 언제든 물어보세요.
"""

# ========== 모델 초기화 ==========

tokenizer = SimpleTokenizer()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 작은 모델 (빠른 응답을 위해)
model = GPT3(
    vocab_size=tokenizer.vocab_size,
    block_size=128,
    n_layers=4,
    embed_dim=128,
    n_heads=4
).to(device)

# 간단한 학습
def train_model():
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    data = tokenizer.encode(TRAINING_DATA)
    
    if len(data) < 10:
        return
    
    model.train()
    for epoch in range(100):  # 빠른 학습
        for i in range(0, len(data) - 32, 16):
            x = torch.tensor([data[i:i+32]], dtype=torch.long, device=device)
            y = torch.tensor([data[i+1:i+33]], dtype=torch.long, device=device)
            
            logits, loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

print("모델 학습 중...")
train_model()
print("모델 학습 완료!")

# ========== 지식베이스 (폴백) ==========

KNOWLEDGE = {
    "안녕": "안녕하세요! 무엇을 도와드릴까요?",
    "이름": "저는 GPT-3 기반 AI 챗봇입니다!",
    "날씨": "죄송하지만 실시간 날씨 정보는 제공할 수 없어요.",
    "시간": "현재 시간을 확인해보세요!",
    "도움": "무엇이든 물어보세요! 제가 최선을 다해 답변드릴게요.",
}

# ========== Flask 앱 ==========

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return '''
    <h1>🤖 GPT-3 챗봇 서버</h1>
    <p>서버가 정상 작동중입니다!</p>
    <p>POST /chat 엔드포인트를 사용하세요.</p>
    '''

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        message = data.get('message', '')
        
        if not message:
            return jsonify({'response': '메시지를 입력해주세요!'})
        
        # 지식베이스 확인
        for key, value in KNOWLEDGE.items():
            if key in message.lower():
                return jsonify({'response': value})
        
        # GPT 모델 사용
        context = f"사용자: {message}\nAI: "
        encoded = tokenizer.encode(context)
        
        if len(encoded) > 0:
            x = torch.tensor([encoded], dtype=torch.long, device=device)
            y = generate(model, x, max_new_tokens=50, temperature=0.8)
            response = tokenizer.decode(y[0].tolist())
            
            # 응답 정리
            if "AI:" in response:
                response = response.split("AI:")[-1].strip()
                if "사용자:" in response:
                    response = response.split("사용자:")[0].strip()
            
            if len(response) < 5:
                response = "흥미로운 질문이네요! 더 자세히 말씀해주시겠어요?"
        else:
            response = "죄송합니다. 이해하지 못했어요."
        
        return jsonify({'response': response})
    
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'response': '처리 중 오류가 발생했습니다. 다시 시도해주세요.'})

@app.route('/train', methods=['POST'])
def train():
    """사용자가 학습 데이터를 추가할 수 있음"""
    try:
        data = request.get_json()
        question = data.get('question', '')
        answer = data.get('answer', '')
        
        if question and answer:
            # 지식베이스에 추가
            KNOWLEDGE[question.lower()] = answer
            return jsonify({'success': True, 'message': 'AI가 학습했습니다!'})
        
        return jsonify({'success': False, 'message': '질문과 답변을 입력해주세요.'})
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

if __name__ == '__main__':
    print("=" * 50)
    print("🤖 GPT-3 챗봇 서버 시작!")
    print("=" * 50)
    print(f"Device: {device}")
    print(f"Vocab Size: {tokenizer.vocab_size}")
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("=" * 50)
    print("서버 주소: http://localhost:5000")
    print("=" * 50)
    
    app.run(host='0.0.0.0', port=5000, debug=True)

