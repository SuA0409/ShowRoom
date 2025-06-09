# 환경 설정 및 import
!pip install torch pandas numpy matplotlib scikit-learn
!pip install transformers sentence-transformers kobert-transformers
!pip install umap-learn hdbscan bertopic
!pip install kss

import torch
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
import hdbscan
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import kss
#---------------------- 리뷰 크롤링 ------------------------
import pandas as pd
from requests import request, get
from requests.compat import urlparse, urlunparse, urljoin
from bs4 import BeautifulSoup
import re
import time
import base64
from urllib.parse import quote, unquote, urlencode
import json
import csv
import requests


# 헤더 정보
headers = {
            "Content-Type": "application/json",
            "User-Agent": "",
            "Accept": "*/*",
            "Referer": "",
            "Accept-Language": "",
            "api-key": "",
        }

# 한국어로 번역한것이 없을 때 or 한국어 댓글이 아닐때 None 반환 함수
def comments(r):
    if r["language"] == 'ko':
        return r["comments"]
    else:
        try:
            return r["localizedReview"]["comments"]
        except:
            return None

# 리뷰에 해당하는 json 호출 함수
def getReviewsJson(stay_id, limit, offset, headers=headers):
    jurl = "" # json url
    variables = {
        "id":stay_id,
        "pdpReviewsRequest":{
            "fieldSelector":"for_p3_translation_only",
            "forPreview":False,
            "limit":limit,
            "offset":str(offset),
            "showingTranslationButton":False,
            "first":limit,
            "sortingPreference":"MOST_RECENT",
            "checkinDate":"2025-06-27",
            "checkoutDate":"2025-07-02",
            "numberOfAdults":"1",
            "numberOfChildren":"0",
            "numberOfInfants":"0",
            "numberOfPets":"0",
            "after":None
        }
    }

    extensions = {
        "persistedQuery":{
            "version":1,
            "sha256Hash":"dec1c8061483e78373602047450322fd474e79ba9afa8d3dbbc27f504030f91d"
        }
    }

    params = {
        "operationName":"StaysPdpReviewsQuery",
        "locale":"ko",
        "currency":"KRW",
        "variables":json.dumps(variables, separators=(',',':')),
        "extensions":json.dumps(extensions, separators=(',',':'))
    }

    resp = requests.get(jurl+urlencode(params, quote_via=quote), headers=headers)
    return resp.json()

# 정규표현식으로 불용어 전처리 함수
def clean_text(text):
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = text.replace('<br>', '').replace('<br/>', '').replace('<br />', '')
    text = re.sub(r'[^\w\s.,$!?가-힣]|[/\\[<a-z>]:\[\]{}|]', '', text)
    text = re.sub(r'[!\"“”‘’?.]+', '. ', text)
    text = re.sub(r'[():]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.strip()


#---------------------- 해당 숙소의 리뷰 추출 ------------------------

url = ''

# ---------- 여기서 url만 가져오면 됨!!! -------------


# 해당 url의 dom을 가져옴
resp = request('get',url)
dom = BeautifulSoup(resp.text, 'html.parser')

p1, p2 = re.split(r'[?]', url)
review = p1+'/reviews?'+p2 # 리뷰 페이지 가는거

url, num = re.split(r'/rooms/', p1) # num은 숙소번호
num = num.split('/reviews')[0] # 진짜 숙소번호
encoding = 'StayListing:'+num # 인코딩 하고 json 호출하기 위한 표현
stay_id = base64.b64encode(encoding.encode('utf-8')).decode('utf-8')

# 최대 50개씩 뽑을 수 있음(넘기면 다른거 가져옴)
limit, cnt = 50, 50
offset = 0
data = []
while 1:
    res_json = getReviewsJson(stay_id, limit, offset)
    try:
		    # 해당 숙소의 리뷰 데이터 접근
        review_data = res_json["data"]["presentation"]["stayProductDetailPage"]["reviews"]
        # 해당 숙소의 리뷰 갯수
        reviewsCount = int(review_data['metadata']['reviewsCount'])
        # 해당 숙소의 리뷰 데이터
        for r in review_data["reviews"]:
            comment_text = comments(r)
            if comment_text:
                data.append({
                    "user_id": r["reviewer"]["id"],
                    "name": r["reviewer"]["firstName"],
                    "stay_id": num,
                    "comment": comment_text,
                    "rating": r.get("rating"),
                    "createdAt": r.get("createdAt")
                })


    except KeyError:
        break
		
		# 마지막 값이 리뷰 최대보다 많다면 무한루프 빠져나옴
    if offset >= reviewsCount:
        break
		
		# 리뷰 최신화
    offset += cnt
    limit += cnt


#---------------------- 리뷰 전처리 ------------------------

# 리뷰 pandas화
reviews = pd.DataFrame(data).dropna().reset_index(drop=True)

rows = [] # 숙소의 문장분리 + 불용어 처리한 리뷰를 담기위한 그릇
for j, row in reviews.iterrows():
    raw_comment = row['comment']
    user_id = row['user_id']
    # 정규식으로 불용어 처리
    cleaned_comment = clean_text(raw_comment)
    # kss로 문장 분리
    sentences = kss.split_sentences(cleaned_comment)

    idx = 0
    split_num = 1

    for sentence in sentences:
        sentence = sentence.strip()
        # 정규식으로 한번 더 문장 분리
        split_sentences = re.split(r'(?<=[가-힣\w])\.(?=[^\d])', sentence)
        if not isinstance(split_sentences, list):
            split_sentences = [sentence]
				
        for s in split_sentences:
            s = s.strip()
            # 초벌한 리뷰가 7이상만 가져감
            if len(s) < 7:
                continue
            start = cleaned_comment.find(s, idx)
            end = start + len(s) - 1
            idx = end + 1
            # rows에 문장 분리한 리뷰를 담음
            rows.append({
                "stay_id": num,
                "user_id": user_id,
                "splitNum": split_num,
                "sentence": s,
                "start": start,
                "end": end
            })
            split_num += 1

# DataFrame 생성
sentence = pd.DataFrame(rows, columns=["stay_id", "user_id", "splitNum", "sentence", "start", "end"])

# 전처리한 리뷰만 가져옴
sentence = sentence['sentence']
# NaN 및 중복 제거하고 list화
docs = sentence.dropna().drop_duplicates().tolist()
# 리뷰 5개 이상만 주제 추출
if len(docs) < 5:
    raise ValueError("데이터가 충분하지 않습니다.")

#---------------------- 모델 사용 ------------------------

# MPS 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 한국어 Sentence-BERT 임베딩 모델 로드 (더 강력한 모델 사용)
embedding_model = SentenceTransformer("jhgan/ko-sbert-nli", device=device)

# 사용자 정의 seed topics 및 이름 (더 구체적이고 데이터 반영)
seed_topics = [
    ["청결도", "깨끗함", "깨끗", "더럽", "드러움", "드럽", "개더럽", "개더러움", "개드럽", "개드러움", "위생", "청소", "더러움", "청결", "불결", "정리", "오염", "깔끔함", "먼지", "청소상태", "냄새", "악취", "향기", "쾌쾌함", "냄새남", "냄새나", "향", "지린내", "곰팡이냄새", "청국장냄새", "냄새문제", "상쾌함", "환기"],
    ["위치", "교통", "가까움", "편리함", "접근성", "원거리", "교통편", "중앙", "외진", "이동", "전망", "근처"],
    ["가격", "비쌈", "저렴함", "가성비", "비용", "고가", "합리적", "경제적", "비싸", "저렴", "요금", "가치", "지불"],
    ["시설", "편리함", "새거", "새것", "새" "편함", "구비", "편의", "시설물", "장비", "부족", "완비", "기능", "설비", "낡음", "오래됨"],
    ["호스트", "서비스", "친절함", "응대", "도움", "불친절", "서비스품질", "관리", "지원", "배려", "체크인", "체크아웃", "입실", "퇴실", "환영", "지연", "빠름", "수속", "접수", "퇴소", "안내"],
    ["소음", "조용함", "시끄러움", "시끄럼", "방음", "소음문제", "고요", "고요함", "소란", "방음효과", "조용", "잡음", "소음원", "방해"],
]
topic_names = ["청결도", "위치", "가격", "시설", "호스트", "소음"]

# UMAP + HDBSCAN 설정 (노이즈 감소)
umap_model = UMAP(n_components=2, random_state=42, metric='cosine', n_neighbors=30, min_dist=0.1)
hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=6, min_samples=3, cluster_selection_method='leaf')

# BERTopic 모델 생성 및 학습
topic_model = BERTopic(
    embedding_model=embedding_model,
    language="multilingual",
    min_topic_size=6,
    nr_topics="auto",
    seed_topic_list=seed_topics,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    verbose=True,
    vectorizer_model=TfidfVectorizer()  # TF-IDF로 키워드 가중치 반영
)

# BERTopic 모델로 돌린 토픽과 확률 가져옴
topics, probs = topic_model.fit_transform(docs)

# 토픽 번호와 키워드 추출
topic_info = topic_model.get_topic_info()
topic_ids = topic_info[topic_info['Topic'] != -1]['Topic'].tolist()

# 토픽 임베딩 기반 유사도 계산 (더 정확한 매핑)
topic_embeddings = topic_model.topic_embeddings_[1:]  # 노이즈 제외
seed_embeddings = embedding_model.encode([" ".join(keywords) for keywords in seed_topics])

similarity = cosine_similarity(topic_embeddings, seed_embeddings)

# 각 토픽을 가장 유사한 seed topic으로 매핑
mapped_topics = {}
for i, sim_row in enumerate(similarity):
    best_idx = sim_row.argmax()
    mapped_topics[topic_ids[i]] = topic_names[best_idx]

selected_labels = topic_names
selected_topic_nums = [k for k, v in mapped_topics.items() if v in selected_labels]
filtered_indices = [i for i, t in enumerate(topics) if t in selected_topic_nums]
filtered_docs = [docs[i] for i in filtered_indices]

# 임베딩 및 UMAP 축소
filtered_embeddings = embedding_model.encode(filtered_docs)
filtered_reduced = umap_model.fit_transform(filtered_embeddings)

# 토픽 이름 기준 문장 분류 (기타 제외)
topic_sentences = defaultdict(list)
for doc, topic_id in zip(docs, topics):
    label = mapped_topics.get(topic_id, "기타")
    if label != "기타":
        topic_sentences[label].append(doc)

# 보기 좋게 출력
for topic_name in topic_names:
    topics = topic_sentences.get(topic_name, [])
    if not topics:
        continue
    print(f"\n토픽: {topic_name}(리뷰 갯수: {len(topics)})")
    for i, topic in enumerate(topics[:5]):  # 최대 5개만 미리보기 출력
        print(f"  {i+1}. {topic}")
    if len(topic) > 5:
        print(f"  ... (총 {len(topics)}개 리뷰 중 5개만 표시)")
