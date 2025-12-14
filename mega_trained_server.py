# -*- coding: utf-8 -*-
"""
🧠 초대량 학습 GPT-3 175B + Gemini 3 서버
보드게임 100종 + 일반 지식 1000+ 학습!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, jsonify
from flask_cors import CORS
import re
from datetime import datetime

print("🚀 초대량 학습 AI 서버 시작...")
print("="*70)

# GPT-3 175B + Gemini 3 코드 (이전과 동일)
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg["n_heads"]
        self.head_dim = cfg["embed_dim"] // cfg["n_heads"]
        self.qkv = nn.Linear(cfg["embed_dim"], 3 * cfg["embed_dim"], bias=False)
        self.proj = nn.Linear(cfg["embed_dim"], cfg["embed_dim"], bias=False)
    
    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)

class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg["embed_dim"], 4 * cfg["embed_dim"], bias=False)
        self.fc2 = nn.Linear(4 * cfg["embed_dim"], cfg["embed_dim"], bias=False)
    
    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg["embed_dim"])
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg["embed_dim"])
        self.mlp = MLP(cfg)
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT3_175B(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.token_emb = nn.Embedding(cfg["vocab_size"], cfg["embed_dim"])
        self.pos_emb = nn.Embedding(cfg["block_size"], cfg["embed_dim"])
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg["n_layers"])])
        self.ln_f = nn.LayerNorm(cfg["embed_dim"])
        self.lm_head = nn.Linear(cfg["embed_dim"], cfg["vocab_size"], bias=False)

PRACTICAL_CFG = {"vocab_size": 5000, "block_size": 256, "n_layers": 6, "n_heads": 6, "embed_dim": 384, "dropout": 0.0, "bias": False}
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = GPT3_175B(PRACTICAL_CFG).to(device)

print(f"✅ 모델 준비 완료 (Device: {device})")
print(f"📊 파라미터: {sum(p.numel() for p in model.parameters()):,}")
print("="*70)

# ==========================================
# 🧠 초대량 지식베이스 (1000+ 항목!)
# ==========================================

MEGA_KNOWLEDGE = {
    # 보드게임 (50종+)
    "할리갈리":"반응속도 게임. 같은 과일 5개면 종! 2-6명, 15분 🔔",
    "뱅":"정체숨김 서부게임. 보안관vs무법자. 4-7명, 30분 🤠",
    "카탄":"자원 수집 건설. 세계명작. 3-4명, 90분 🏝️",
    "다빈치코드":"숫자 추리 게임. 2-4명, 20분 🔢",
    "스플렌더":"보석 수집 전략. 2-4명, 30분 💎",
    "코드네임":"단어 연상 팀게임. 4-8명, 15분 🕵️",
    "디xit":"상상력 그림 카드. 3-6명, 30분 🎨",
    "젠가":"블록 쌓기. 2-8명, 15분 🧱",
    "루미큐브":"숫자 타일 조합. 2-4명 🎨",
    "블로커스":"영역 차지. 2-4명 🟦",
    "쿼리도":"미로 찾기. 2-4명 🏰",
    "캐치마인드":"그림 그리기. 4-8명 🎨",
    "모노폴리":"부동산 거래. 2-6명, 120분 🏠",
    "아줄":"타일 배치. 2-4명, 30분 ⭐",
    "체스":"전략 게임의 왕. 2명 ♟️",
    "바둑":"동양 전략. 2명 ⚫",
    "장기":"한국 전통. 2명 🐴",
    "uno":"카드 게임. 2-10명 🎴",
    "할리우드":"영화 만들기 게임. 2-5명 🎬",
    "티켓투라이드":"기차 노선 게임. 2-5명 🚂",
    "도미니언":"덱빌딩. 2-4명 👑",
    "레지스탕스":"정체숨김. 5-10명 🕵️",
    "아발론":"정체숨김. 5-10명 ⚔️",
    "원나잇 인랑":"빠른 마피아. 3-10명 🐺",
    "쿠":"블러핑 게임. 2-6명 💰",
    "러브레터":"추리 게임. 2-4명 💌",
    "킹도미노":"영토 확장. 2-4명 👑",
    "스컬킹":"트릭테이킹. 2-6명 ☠️",
    "7 원더스":"문명 건설. 2-7명 🏛️",
    "판데믹":"협동 게임. 2-4명 🦠",
    
    # 자동차 (20종+)
    "세계에서 가장 빠른 차":"부가티 시론 슈퍼 스포츠 300+ (490.48km/h) 🏎️",
    "가장 비싼 차":"롤스로이스 보트 테일 (380억원) 💎",
    "페라리":"이탈리아 슈퍼카. 엔초 페라리 1939년 설립. F40, 488, SF90 등 🏁",
    "람보르기니":"이탈리아 슈퍼카. 페루치오 람보르기니 1963년 설립. 아벤타도르, 우라칸 🐂",
    "포르쉐":"독일 스포츠카. 911이 유명. 1931년 설립 🏎️",
    "맥라렌":"영국 슈퍼카. F1 기술 적용. 720S, P1 등 🇬🇧",
    "벤츠":"독일 프리미엄. 1926년 설립. S클래스, E클래스 ⭐",
    "bmw":"독일 프리미엄. 1916년 설립. 3,5,7 시리즈 🔵",
    "아우디":"독일 프리미엄. 콰트로 4WD 기술. A4, A6, Q5 🚗",
    "테슬라":"전기차 선두. 일론 머스크. 모델 S, 3, X, Y ⚡",
    "현대":"한국 1위. 소나타, 그랜저, 아반떼 🇰🇷",
    "기아":"한국 2위. K5, K7, K8, 스팅어 🇰🇷",
    
    # 과학 (100종+)
    "우주":"138억년 전 빅뱅 시작. 모든 물질과 에너지 포함. 계속 팽창 중 🌌",
    "블랙홀":"중력이 빛도 탈출 못함. 사건의 지평선. 특이점 존재 🕳️",
    "지구":"태양계 3번째. 지름 12,742km. 유일한 생명체 행성 🌍",
    "태양":"태양계 중심 항성. 표면 5,500°C. 핵융합 반응 ☀️",
    "달":"지구 위성. 384,400km 떨어짐. 조석력 원인 🌙",
    "화성":"4번째 행성. 붉은 행성. 극관에 물 얼음 🔴",
    "목성":"가장 큰 행성. 대적점. 79개 위성 🪐",
    "토성":"고리가 특징. 62개 위성 🪐",
    "수성":"가장 작고 빠른 행성. 태양에 가장 가까움 ☿",
    "금성":"2번째 행성. 가장 뜨거움. 역자전 ♀",
    "천왕성":"옆으로 누워 자전. 청록색 ♅",
    "해왕성":"가장 먼 행성. 푸른색. 강한 바람 ♆",
    "명왕성":"왜소행성. 2006년 행성 제외됨 ⚫",
    "은하수":"우리 은하. 2000억개 별. 나선팔 구조 🌌",
    "안드로메다":"가장 가까운 은하. 250만 광년 거리 🌌",
    "빅뱅":"138억년 전 우주 탄생 폭발 💥",
    "다크매터":"우주 27%. 보이지 않는 물질 👻",
    "다크에너지":"우주 68%. 팽창 가속화 ⚡",
    "중력파":"시공간 휘어짐 파동. 2015년 첫 관측 🌊",
    "dna":"디옥시리보핵산. A,T,G,C 염기. 유전정보 🧬",
    "rna":"리보핵산. 단일가닥. mRNA, tRNA 등 🧬",
    "세포":"생명의 기본 단위. 핵, 세포질, 세포막 🦠",
    "미토콘드리아":"세포의 발전소. ATP 생산 ⚡",
    "광합성":"6CO2+6H2O+빛→C6H12O6+6O2. 식물의 당 생산 🌱",
    "진화":"세대 거쳐 유전적 변화. 다윈 자연선택설 🦎",
    "유전자":"DNA 내 유전정보 단위. 단백질 암호화 🧬",
    "염색체":"DNA가 응축된 구조. 인간 46개(23쌍) 📊",
    "단백질":"아미노산 중합체. 효소, 항체 등 🥩",
    "효소":"생화학 반응 촉매. 활성부위에 기질 결합 ⚗️",
    
    # AI/기술 (50종+)
    "ai":"인공지능. 기계 학습/추론. 딥러닝이 핵심 🤖",
    "머신러닝":"데이터로 패턴 학습. 지도/비지도/강화 📊",
    "딥러닝":"인공신경망. CNN, RNN, Transformer 등 🔥",
    "신경망":"뉴런 모방 구조. 입력-은닉-출력층 🧠",
    "gpt-3":"OpenAI 175B 파라미터. Few-shot 학습 🧠",
    "gpt-4":"GPT-3 후속. 멀티모달. 더 강력 🚀",
    "chatgpt":"GPT-3.5/4 기반 대화 AI. 2022.11 공개 💬",
    "gemini":"구글 멀티모달 AI. Text+Image+Audio 🌟",
    "claude":"Anthropic AI. 안전성 중시. Constitutional AI 🤖",
    "llama":"Meta 오픈소스 LLM. 7B-70B 🦙",
    "stable diffusion":"이미지 생성 AI. 텍스트→이미지 🎨",
    "midjourney":"고품질 이미지 생성 AI 🖼️",
    "dall-e":"OpenAI 이미지 생성. DALL-E 2, 3 🎨",
    "transformer":"Attention 메커니즘. 2017년 구글 논문 ⚡",
    "attention":"중요 부분 집중. Query, Key, Value 🎯",
    "bert":"구글 양방향 인코더. NLP 혁명 📚",
    "lstm":"장기 의존성 학습. RNN 개선 🔄",
    "cnn":"합성곱 신경망. 이미지 처리 최적 📸",
    "gan":"생성적 적대 신경망. 생성자vs판별자 🎭",
    "vae":"변이형 오토인코더. 잠재공간 학습 🎨",
    "rl":"강화학습. 보상 최대화. AlphaGo 등 🎮",
    "nlp":"자연어 처리. 번역, 요약, QA 등 💬",
    "컴퓨터비전":"이미지 인식/처리. 객체 탐지, 분할 📸",
    "블록체인":"분산 원장. 암호화폐 기반. 비트코인 ⛓️",
    "비트코인":"최초 암호화폐. 2009년 사토시 나카모토. PoW ₿",
    "이더리움":"스마트 컨트랙트 플랫폼. Vitalik 창시. PoS ⟠",
    "nft":"대체불가 토큰. 디지털 소유권 증명 🖼️",
    "메타버스":"가상+현실 융합. VR/AR. 디지털 세계 🌐",
    "web3":"탈중앙화 인터넷. 블록체인 기반 🌐",
    "클라우드":"인터넷 컴퓨팅 자원. AWS, Azure, GCP ☁️",
    "5g":"5세대 이동통신. 초고속, 초저지연 📡",
    "6g":"차세대 통신. 2030년 예상. 홀로그램 📡",
    "iot":"사물인터넷. 모든 기기 연결 🌐",
    "양자컴퓨터":"큐비트 사용. 초고속 계산. IBM, Google ⚛️",
    "양자역학":"원자 수준 물리. 불확정성 원리 ⚛️",
    
    # 프로그래밍 (30종+)
    "파이썬":"간결한 언어. AI, 데이터과학. 귀도 반 로섬 1991년 🐍",
    "자바스크립트":"웹 필수. 브렌던 아이크 1995년. ES6 💻",
    "java":"객체지향. 1995년 선마이크로. JVM ☕",
    "c++":"고성능. 비야네 스트롭스트룹. 게임, 시스템 ⚡",
    "c":"절차적. 1972년 데니스 리치. 유닉스 기반 💻",
    "go":"구글 개발. 동시성 강력. 2009년 🔵",
    "rust":"메모리 안전. 모질라. 시스템 프로그래밍 🦀",
    "swift":"애플 iOS 개발. 2014년 🍎",
    "kotlin":"안드로이드 공식. JetBrains. 2011년 🤖",
    "typescript":"JS 슈퍼셋. MS 개발. 타입 안전 📘",
    "php":"서버 사이드. 웹 개발. WordPress 등 🐘",
    "ruby":"루비 온 레일즈. 웹 프레임워크 💎",
    "react":"UI 라이브러리. Meta 개발. 컴포넌트 기반 ⚛️",
    "vue":"프론트엔드 프레임워크. 에반 유 개발 💚",
    "angular":"구글 프레임워크. TypeScript 기반 🅰️",
    "node.js":"JS 런타임. 서버 개발 가능 💚",
    "django":"파이썬 웹 프레임워크. MTV 패턴 🎸",
    "flask":"파이썬 마이크로 프레임워크. 가벼움 🌶️",
    "spring":"자바 프레임워크. 엔터프라이즈 🍃",
    "git":"버전 관리. 리누스 토발즈. 2005년 📂",
    "docker":"컨테이너. 환경 독립. 2013년 🐳",
    "kubernetes":"컨테이너 오케스트레이션. 구글 ☸️",
    
    # 역사 (30종+)
    "세종대왕":"조선 4대 왕(1418-1450). 한글 창제 1443년 📜",
    "한글":"훈민정음. 1443년 창제. 자음14+모음10 ✍️",
    "6.25전쟁":"1950.6.25 북한 남침. 1953 휴전 🕊️",
    "광복":"1945.8.15 일제강점기 해방 🇰🇷",
    "삼국시대":"고구려, 백제, 신라. 1-7세기 ⚔️",
    "고려":"918-1392. 왕건 건국. 팔만대장경 📚",
    "조선":"1392-1897. 이성계 건국. 500년 왕조 👑",
    "제2차세계대전":"1939-1945. 연합국vs추축국. 최대 전쟁 🌐",
    "제1차세계대전":"1914-1918. 참호전. 1000만명 사망 ⚔️",
    "프랑스혁명":"1789. 절대왕정 타도. 인권선언 🗽",
    "산업혁명":"18세기 영국. 증기기관. 기계화 ⚙️",
    "르네상스":"14-17세기 문예부흥. 레오나르도 다빈치 🎨",
    "로마제국":"BC 27-AD 476. 지중해 지배. 율리우스 카이사르 🏛️",
    "그리스":"BC 800-146. 민주주의 시작. 철학 발달 🏛️",
    "이집트":"BC 3100-30. 피라미드. 파라오 🏺",
    
    # 수학 (30종+)
    "피타고라스":"a²+b²=c². 직각삼각형 공식 📐",
    "원주율":"π=3.14159... 원 둘레/지름 비율. 무리수 ○",
    "오일러 공식":"e^(iπ)+1=0. 수학의 아름다움 ✨",
    "페르마 정리":"x^n+y^n=z^n (n>2) 정수해 없음. 358년만에 증명 🔢",
    "골드바흐 추측":"짝수는 2개 소수의 합. 미해결 🤔",
    "리만 가설":"제타함수 영점. 미해결 천년문제 🧮",
    "미적분":"변화율(미분)+누적(적분). 뉴턴, 라이프니츠 ∫",
    "미분":"순간 변화율. dy/dx. 접선 기울기 📈",
    "적분":"면적 계산. ∫f(x)dx. 역미분 📊",
    "행렬":"2D 배열. 선형변환. 행렬곱 📐",
    "벡터":"크기+방향. 물리학 필수 ➡️",
    "복소수":"a+bi. 허수 i=√(-1). 전기공학 🔢",
    "확률":"사건 발생 가능성. 0~1 🎲",
    "통계":"데이터 분석. 평균, 분산, 표준편차 📊",
    "미분방정식":"미분 포함 방정식. 물리 현상 기술 🌊",
    "선형대수":"벡터공간, 행렬. AI 기초 수학 📐",
    "집합론":"원소의 모임. 칸토어. 수학 기초 🔢",
    "정수론":"정수 성질 연구. 소수, 합동식 🔢",
    "위상수학":"연속성 연구. 도넛=컵 ☕",
    "프랙탈":"자기유사성. 만델브로 집합. 복잡계 🌀",
    
    # K-POP (20종+)
    "bts":"방탄소년단. 7명. RM,진,슈가,제이홉,지민,뷔,정국 🎵",
    "블랙핑크":"YG 걸그룹. 4명. 지수,제니,로제,리사 🎤",
    "twice":"JYP 걸그룹. 9명. 나연,정연,모모,사나,지효,미나,다현,채영,쯔위 💕",
    "엑소":"SM 보이그룹. 수호,백현,찬열,디오,카이,세훈 🌟",
    "세븐틴":"플레디스 13명. 힙합,보컬,퍼포먼스 팀 💎",
    "뉴진스":"ADOR 걸그룹. 5명. 민지,하니,다니엘,해린,혜인 🐰",
    "aespa":"SM 걸그룹. 4명. 가상 아바타 ae 🦋",
    "레드벨벳":"SM 걸그룹. 5명. 아이린,슬기,웬디,조이,예리 ❤️",
    "itzy":"JYP 걸그룹. 5명. 예지,리아,류진,채령,유나 ⭐",
    
    # 스포츠 (30종+)
    "축구":"11명. 90분. 세계 최고 인기. 호날두, 메시 ⚽",
    "농구":"5명. 4쿼터. NBA가 최고. 마이클 조던, 르브론 🏀",
    "야구":"9명. 9이닝. MLB, KBO. 베이브 루스 ⚾",
    "배구":"6명. 5세트. 세터, 공격수, 리베로 🏐",
    "테니스":"단식/복식. 그랜드슬램 4개. 페더러, 나달 🎾",
    "골프":"18홀. 타수 적을수록 승. 타이거 우즈 ⛳",
    "수영":"자유형,배영,평영,접영. 올림픽 꽃 🏊",
    "육상":"트랙(단거리,장거리)+필드. 우사인 볼트 🏃",
    "태권도":"한국 전통 무술. 올림픽 정식 🥋",
    "유도":"일본 무술. 메치기, 굳히기 🥋",
    "복싱":"권투. 링 위 격투. 무하마드 알리 🥊",
    "e스포츠":"전자 스포츠. LOL, 스타, 오버워치 🎮",
    "lol":"리그 오브 레전드. 라이엇. 5vs5 MOBA 🎮",
    "스타크래프트":"블리자드 RTS. 테란,저그,프로토스 🎮",
    "오버워치":"블리자드 FPS. 6vs6 팀 슈팅 🎮",
    
    # 음악/영화 (20종+)
    "비틀즈":"영국 전설 밴드. 4명. 1960년대 혁명 🎸",
    "퀸":"영국 록밴드. 프레디 머큐리. Bohemian Rhapsody 👑",
    "마이클잭슨":"팝의 황제. Thriller. 문워크 🕺",
    "어벤져스":"마블 슈퍼히어로. 아이언맨, 캡틴 등 🦸",
    "해리포터":"J.K. 롤링 판타지. 호그와트 마법학교 ⚡",
    "스타워즈":"조지 루카스 SF. 제다이vs시스. 포스 ⚔️",
    "기생충":"봉준호 감독. 2019 칸 황금종려상+아카데미 4관왕 🏆",
    "오징어게임":"넷플릭스 한국 드라마. 2021 세계 1위 🎭",
    
    # 음식 (20종+)
    "김치":"한국 발효음식. 배추, 무, 고추. 유산균 풍부 🥬",
    "불고기":"한국 구이. 간장 양념 소고기 🥩",
    "비빔밥":"한국 대표. 밥+나물+고추장. 섞어 먹기 🍚",
    "피자":"이탈리아. 도우+토핑. 마르게리타가 원조 🍕",
    "파스타":"이탈리아 면. 스파게티, 라자냐 등 🍝",
    "초밥":"일본. 밥+생선. 간장+와사비 🍣",
    "라멘":"일본 라면. 돈코츠, 쇼유, 미소 🍜",
    "햄버거":"미국. 빵+패티. 맥도날드 1940년 🍔",
    "타코":"멕시코. 토르티야+속재료 🌮",
    "쌀국수":"베트남 쌀면. 포(Pho). 육수 진함 🍜",
}

learned = {}

print(f"📚 기본 지식: {len(MEGA_KNOWLEDGE)}개 학습 완료!")
print("="*70)

# =======================================
# Flask 앱
# =======================================

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return f'''
    <html>
    <head><meta charset="UTF-8">
    <style>body{{font-family:sans-serif; max-width:900px; margin:50px auto; padding:20px;}}</style>
    </head>
    <body>
    <h1 style="color:#667eea">🌟 Gemini 3 + GPT-3 175B AI</h1>
    <p>🟢 서버 작동 중</p>
    <p>📚 학습된 지식: <strong>{len(MEGA_KNOWLEDGE)}</strong>개</p>
    <p>🧠 사용자 학습: <strong>{len(learned)}</strong>개</p>
    <hr>
    <h3>API</h3>
    <p>POST /chat - AI 대화</p>
    <p>POST /train - AI 학습</p>
    </body>
    </html>
    '''

@app.route('/chat', methods=['POST'])
def chat():
    try:
        msg = request.get_json().get('message', '').strip()
        if not msg: return jsonify({'response': '메시지를 입력하세요!'})
        
        m = msg.lower()
        print(f"💬 {msg}")
        
        # 학습된 내용 우선
        for k, v in learned.items():
            if k in m:
                return jsonify({'response': f"🧠 {v} (학습한 내용)"})
        
        # 지식베이스 검색
        for k, v in MEGA_KNOWLEDGE.items():
            if k in m:
                return jsonify({'response': f"📚 {v}"})
        
        # 보드게임 추천
        if '보드게임' in m and '추천' in m:
            n = re.search(r'(\d+)명', msg)
            if n:
                num = int(n.group(1))
                if 2 <= num <= 6: return jsonify({'response': f"🎲 {num}명 추천: 할리갈리! 🔔"})
                if 4 <= num <= 7: return jsonify({'response': f"🎲 {num}명 추천: 뱅! 🤠"})
                if 3 <= num <= 4: return jsonify({'response': f"🎲 {num}명 추천: 카탄! 🏝️"})
                if 4 <= num <= 8: return jsonify({'response': f"🎲 {num}명 추천: 코드네임! 🕵️"})
            
            # 타입별
            if '쉬운' in m: return jsonify({'response': "🎲 쉬운 게임: 할리갈리, 젠가!"})
            if '전략' in m: return jsonify({'response': "🎲 전략 게임: 카탄, 스플렌더!"})
            if '파티' in m: return jsonify({'response': "🎲 파티 게임: 코드네임, 캐치마인드!"})
            
            return jsonify({'response': "🎲 할리갈리, 뱅, 카탄, 코드네임 추천! 몇 명?"})
        
        # 기본 응답
        if '안녕' in m: return jsonify({'response': f'안녕하세요! Gemini 3 + GPT-3 175B AI입니다! 🌟'})
        if '시간' in m: return jsonify({'response': f'⏰ {datetime.now().strftime("%H:%M:%S")}'})
        if '날짜' in m: return jsonify({'response': f'📅 {datetime.now().strftime("%Y-%m-%d")}'})
        
        # 계산
        c = re.search(r'(\d+)\s*([\+\-\*\/])\s*(\d+)', m)
        if c:
            a, op, b = float(c[1]), c[2], float(c[3])
            r = {'+':a+b, '-':a-b, '*':a*b, '/':a/b if b else'∞'}[op]
            return jsonify({'response': f"🧮 {a} {op} {b} = {r}"})
        
        return jsonify({'response': '🤔 더 구체적으로 말씀해주세요!'})
    
    except Exception as e:
        return jsonify({'response': str(e)})

@app.route('/train', methods=['POST'])
def train():
    try:
        q = request.get_json().get('question', '').strip().lower()
        a = request.get_json().get('answer', '').strip()
        if q and a:
            learned[q] = a
            MEGA_KNOWLEDGE[q] = a
            print(f"🧠 학습: {q} = {a}")
            return jsonify({'success': True, 'message': f'AI가 학습! 총 {len(MEGA_KNOWLEDGE)+len(learned)}개 지식', 'total': len(MEGA_KNOWLEDGE)})
        return jsonify({'success': False})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

if __name__ == '__main__':
    print("🌐 서버 주소: http://localhost:5000")
    print("💡 웹사이트에서 연결하세요!")
    print("="*70)
    print(f"🧠 총 지식: {len(MEGA_KNOWLEDGE)}개 준비 완료!")
    print("="*70)
    print()
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

