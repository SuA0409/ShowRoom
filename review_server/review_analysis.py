"""
 review_analysis.py
- Airbnb 숙소의 리뷰를 수집하고 문장 단위로 분해
- BERTopic으로 토픽 모델링 및 주제별 문장 분류
"""

import base64, re, requests, json
import pandas as pd
from bs4 import BeautifulSoup
from collections import defaultdict

# 모델 및 분석 도구
import torch
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
import hdbscan
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import kss
from urllib.parse import quote, urlencode


# ==========================================
#  한글 리뷰 혹은 번역본 추출 함수
# ==========================================
def comments(r):
    """
    한글 리뷰 또는 번역본 리뷰를 반환
    """
    if r["language"] == 'ko':
        return r["comments"]
    else:
        try:
            return r["localizedReview"]["comments"]
        except:
            return None


# ==========================================
#  Airbnb 리뷰 API 요청 함수
# ==========================================
headers = {
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0",
    "Accept": "*/*",
    "Referer": "https://www.airbnb.co.kr/",
    "Accept-Language": "ko-KR,ko;q=0.9",
    "x-airbnb-api-key": "d306zoyjsyarp7ifhu67rjxn52tv0t20",
}

def getReviewsJson(stay_id, limit, offset, headers=headers):
    """
    Airbnb API를 통해 JSON 형식의 리뷰 데이터 수집
    """
    jurl = "https://www.airbnb.co.kr/api/v3/StaysPdpReviewsQuery/dec1c8061483e78373602047450322fd474e79ba9afa8d3dbbc27f504030f91d?"
    variables = {
        "id": stay_id,
        "pdpReviewsRequest": {
            "fieldSelector": "for_p3_translation_only",
            "forPreview": False,
            "limit": limit,
            "offset": str(offset),
            "showingTranslationButton": False,
            "first": limit,
            "sortingPreference": "MOST_RECENT",
            "checkinDate": "2025-06-27",
            "checkoutDate": "2025-07-02",
            "numberOfAdults": "1",
            "numberOfChildren": "0",
            "numberOfInfants": "0",
            "numberOfPets": "0",
            "after": None
        }
    }
    extensions = {
        "persistedQuery": {
            "version": 1,
            "sha256Hash": "dec1c8061483e78373602047450322fd474e79ba9afa8d3dbbc27f504030f91d"
        }
    }
    params = {
        "operationName": "StaysPdpReviewsQuery",
        "locale": "ko",
        "currency": "KRW",
        "variables": json.dumps(variables, separators=(',', ':')),
        "extensions": json.dumps(extensions, separators=(',', ':'))
    }
    resp = requests.get(jurl + urlencode(params, quote_via=quote), headers=headers)
    return resp.json()


# ==========================================
#  리뷰 텍스트 전처리 함수
# ==========================================
def clean_text(text):
    """
    텍스트에서 특수문자, 태그, 개행문자 제거 및 정리
    """
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'<br.*?>', ' ', text)
    text = re.sub(r'[^\w\s.,!?가-힣]', '', text)
    text = re.sub(r'([.!?])', r'\1 ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ==========================================
#  숙소 URL 기반 리뷰 분석 실행
# ==========================================
def run_topic_model_on_room(room_url):
    """
    1️⃣ 숙소 ID 추출 및 인코딩
    2️⃣ Airbnb API로 리뷰 수집
    3️⃣ 문장 단위 분해 및 전처리
    4️⃣ BERTopic으로 토픽 모델링 및 유사도 기반 매핑
    5️⃣ 주제별 문장 분류 결과 반환
    """
    # 1️⃣ 숙소 ID 인코딩
    p1, _ = room_url.split("?")
    _, num = p1.split("/rooms/")
    num = num.split("/")[0]
    encoding = 'StayListing:' + num
    stay_id = base64.b64encode(encoding.encode('utf-8')).decode('utf-8')

    # 2️⃣ 리뷰 수집
    data = []
    offset = 0
    limit = 50
    while True:
        res_json = getReviewsJson(stay_id, limit, offset)
        try:
            review_data = res_json["data"]["presentation"]["stayProductDetailPage"]["reviews"]
            reviewsCount = int(review_data['metadata']['reviewsCount'])
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
        except:
            break
        if offset >= reviewsCount:
            break
        offset += 50
        limit += 50

    # 3️⃣ 리뷰 DataFrame 생성 및 검증
    reviews = pd.DataFrame(data).dropna()
    if len(reviews) < 5:
        raise ValueError("리뷰가 충분하지 않습니다.")

    # 4️⃣ 문장 단위로 분해
    rows = []
    for j, row in reviews.reset_index(drop=True).iterrows():
        cleaned_comment = clean_text(row['comment'])
        sentences = kss.split_sentences(cleaned_comment)
        idx, split_num = 0, 1
        for sentence in sentences:
            for s in re.split(r'(?<=[가-힣\w])\.(?=[^\d])', sentence.strip()):
                s = s.strip()
                if len(s) < 7: continue
                start = cleaned_comment.find(s, idx)
                end = start + len(s) - 1
                idx = end + 1
                rows.append({
                    "stay_id": num,
                    "user_id": row['user_id'],
                    "splitNum": split_num,
                    "sentence": s,
                    "start": start,
                    "end": end
                })
                split_num += 1

    # 5️⃣ 고유 문장 리스트 준비
    sentence_df = pd.DataFrame(rows).dropna()
    docs = sentence_df['sentence'].drop_duplicates().tolist()
    if len(docs) < 5:
        raise ValueError("문장 수가 너무 적습니다.")

    # 6️⃣ 임베딩 모델 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embedding_model = SentenceTransformer("jhgan/ko-sbert-nli", device=device)

    # 7️⃣ 주제별 시드 키워드 & 이름 정의
    seed_topics = [
        ["청결도", "깨끗함", "위생", "청소", "정리", "깔끔", "냄새", "쾌쾌함", "곰팡이냄새", "환기"],
        ["위치", "교통", "편리함", "접근성", "중앙", "근처"],
        ["가격", "저렴", "가성비", "비용", "고가", "경제적", "요금", "가치"],
        ["시설", "편리함", "새것", "구비", "장비", "기능", "설비", "낡음"],
        ["호스트", "서비스", "친절", "응대", "도움", "불친절", "관리", "체크인", "체크아웃"],
        ["소음", "조용", "시끄러움", "방음", "소란", "방해"]
    ]
    topic_names = ["청결도", "소음", "위치", "가격", "시설", "호스트"]

    # 8️⃣ 차원 축소 및 클러스터링 모델 정의
    umap_model = UMAP(n_components=2, random_state=42, metric='cosine', n_neighbors=30, min_dist=0.1)
    hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=6, min_samples=3, cluster_selection_method='leaf')

    # 9️⃣ BERTopic 모델 학습
    topic_model = BERTopic(
        embedding_model=embedding_model,
        language="multilingual",
        min_topic_size=6,
        nr_topics="auto",
        seed_topic_list=seed_topics,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        verbose=True,
        vectorizer_model=TfidfVectorizer()
    )
    topics, probs = topic_model.fit_transform(docs)
    topic_info = topic_model.get_topic_info()
    topic_ids = topic_info[topic_info['Topic'] != -1]['Topic'].tolist()

    # 🔟 Cosine Similarity로 토픽 이름 매핑
    topic_embeddings = topic_model.topic_embeddings_[1:]
    seed_embeddings = embedding_model.encode([" ".join(keywords) for keywords in seed_topics])
    similarity = cosine_similarity(topic_embeddings, seed_embeddings)
    mapped_topics = {topic_ids[i]: topic_names[sim_row.argmax()] for i, sim_row in enumerate(similarity)}

    # 🔟 문장들을 주제별로 분류
    topic_sentences = defaultdict(list)
    for doc, topic_id in zip(docs, topics):
        label = mapped_topics.get(topic_id, "기타")
        if label != "기타":
            topic_sentences[label].append(doc)

    # 🔟 최종 결과 반환
    return {"stay_id": num, "topics": topic_sentences}
