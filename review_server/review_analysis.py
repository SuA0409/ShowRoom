"""
 review_analysis.py
- Airbnb 숙소의 리뷰를 수집하고 문장 단위로 분해
- BERTopic으로 토픽 모델링 및 주제별 문장 분류
"""
import base64
import re
import requests
import json
import pandas as pd
from bs4 import BeautifulSoup
from collections import defaultdict
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

# ==========================================
#  한글 리뷰 혹은 번역본 추출 함수
# ==========================================
def comments(korean):
    """한국어 리뷰 또는 번역된 리뷰를 추출하는 함수

    Args:
        korean (dict): 리뷰 데이터를 포함한 딕셔너리

    Returns:
        str or None: 한국어 리뷰 텍스트, 번역된 리뷰 텍스트, 또는 조건에 맞지 않으면 None
    """
    if korean["language"] == 'ko':
        return korean["comments"]
    else:
        try:
            return korean["localizedReview"]["comments"]
        except KeyError:
            return None

def getReviewsJson(stay_id, limit, offset, headers=headers):
    """Airbnb API에서 숙소 리뷰 JSON 데이터를 가져오는 함수

    Args:
        stay_id (str): Base64로 인코딩된 숙소 ID
        limit (int): 한 번에 가져올 리뷰 수
        offset (int): 리뷰 조회 시작 지점
        headers (dict, optional): API 요청에 필요한 헤더 정보. 기본값은 전역 headers

    Returns:
        dict: 리뷰 데이터가 포함된 JSON 응답
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
    """리뷰 텍스트를 정제하여 불용어와 특수문자를 제거하는 함수

    Args:
        text (str): 전처리할 원본 리뷰 텍스트

    Returns:
        str: 불용어와 특수문자가 제거된 정제된 텍스트
    """
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = text.replace('<br>', '').replace('<br/>', '').replace('<br />', '')
    text = re.sub(r'[^\w\s.,$!?가-힣]|[/\\[<a-z>]:\[\]{}|]', '', text)
    text = re.sub(r'[!\"“”‘’?.]+', '. ', text)
    text = re.sub(r'[():]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.strip()

# ==========================================
#  숙소 URL 기반 리뷰 분석 실행
# ==========================================
def run_topic_model_on_room(room_url):
    """숙소 URL을 기반으로 리뷰를 수집하고 토픽 모델링 분석을 수행하는 함수

    Args:
        room_url (str): Airbnb 숙소 URL

    Returns:
        dict: 숙소 ID와 주제별 리뷰 문장이 포함된 딕셔너리

    Raises:
        ValueError: 리뷰 문장이 5개 미만일 경우
    """
    # 숙소 ID 추출 및 stay_id 인코딩
    p1, _ = room_url.split("?")
    _, num = p1.split("/rooms/")
    num = num.split("/")[0]
    encoding = 'StayListing:' + num
    stay_id = base64.b64encode(encoding.encode('utf-8')).decode('utf-8')

    # 리뷰 수집
    limit, cnt, offset = 50, 50, 0
    data = []
    while True:
        res_json = getReviewsJson(stay_id, limit, offset, headers)
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
                        "rating": r["rating"],
                        "createdAt": r["createdAt"]
                    })
        except KeyError:
            break

        if offset >= reviewsCount:
            break
        offset += cnt
        limit += cnt

    # 리뷰 DataFrame 생성 및 검증
    reviews = pd.DataFrame(data).dropna().reset_index(drop=True)
    rows = []
    for j, row in reviews.iterrows():
        raw_comment = row['comment']
        user_id = row['user_id']
        cleaned_comment = clean_text(raw_comment)
        sentences = kss.split_sentences(cleaned_comment)
        idx = 0
        split_num = 1

        for sentence in sentences:
            sentence = sentence.strip()
            split_sentences = re.split(r'(?<=[가-힣\w])\.(?=[^\d])', sentence)
            if not isinstance(split_sentences, list):
                split_sentences = [sentence]

            for s in split_sentences:
                s = s.strip()
                if len(s) < 7:
                    continue
                start = cleaned_comment.find(s, idx)
                end = start + len(s) - 1
                idx = end + 1
                rows.append({
                    "stay_id": num,
                    "user_id": user_id,
                    "splitNum": split_num,
                    "sentence": s,
                    "start": start,
                    "end": end
                })
                split_num += 1

    sentence = pd.DataFrame(rows, columns=["stay_id", "user_id", "splitNum", "sentence", "start", "end"])
    sentence = sentence['sentence']
    docs = sentence.dropna().drop_duplicates().tolist()

    if len(docs) < 5:
        raise ValueError("데이터가 충분하지 않습니다.")

    # 임베딩 모델 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embedding_model = SentenceTransformer("jhgan/ko-sbert-nli", device=device)

    # 주제별 시드 키워드 정의 및 주제 이름 정의
    seed_topics = [
        ["청결도", "깨끗함", "깨끗", "더럽", "드러움", "드럽", "개더럽", "개드러움", "개드럽", "개드러움", "위생", "청소", "더러움", "청결", "불결", "정리", "오염", "깔끔함", "먼지", "청소상태", "냄새", "악취", "향기", "쾌쾌함", "냄새남", "냄새나", "향", "지린내", "곰팡이냄새", "청국장냄새", "냄새문제", "상쾌함", "환기"],
        ["소음", "조용함", "시끄러움", "시끄럼", "방음", "소음문제", "고요", "고요함", "소란", "방음효과", "조용", "잡음", "소음원", "방해"],
        ["위치", "교통", "가까움", "편리함", "접근성", "원거리", "교통편", "중앙", "외진", "이동", "전망", "근처"],
        ["가격", "비쌈", "저렴함", "가성비", "비용", "고가", "합리적", "경제적", "비싸", "저렴", "요금", "가치", "지불"],
        ["시설", "편리함", "새거", "새것", "새", "편함", "구비", "편의", "시설물", "장비", "부족", "완비", "기능", "설비", "낡음", "오래됨"],
        ["호스트", "서비스", "친절함", "응대", "도움", "불친절", "서비스품질", "관리", "지원", "배려", "체크인", "체크아웃", "입실", "퇴실", "환영", "지연", "빠름", "수속", "접수", "퇴소", "안내"],
    ]
    topic_names = ["청결도", "소음", "위치", "가격", "시설", "호스트"]

    umap_model = UMAP(n_components=2, random_state=42, metric='cosine', n_neighbors=30, min_dist=0.1)
    hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=6, min_samples=3, cluster_selection_method='leaf')
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

    # Cosine Similarity로 토픽 이름 매핑
    topic_embeddings = topic_model.topic_embeddings_[1:]
    seed_embeddings = embedding_model.encode([" ".join(keywords) for keywords in seed_topics])
    similarity = cosine_similarity(topic_embeddings, seed_embeddings)

    mapped_topics = {}
    for i, sim_row in enumerate(similarity):
        best_idx = sim_row.argmax()
        mapped_topics[topic_ids[i]] = topic_names[best_idx]

    selected_labels = topic_names
    selected_topic_nums = [k for k, v in mapped_topics.items() if v in selected_labels]
    filtered_indices = [i for i, t in enumerate(topics) if t in selected_topic_nums]
    filtered_docs = [docs[i] for i in filtered_indices]

    filtered_embeddings = embedding_model.encode(filtered_docs)
    filtered_reduced = umap_model.fit_transform(filtered_embeddings)

    # 문장들을 주제별로 분류
    topic_sentences = defaultdict(list)
    for doc, topic_id in zip(docs, topics):
        label = mapped_topics.get(topic_id, "기타")
        if label != "기타":
            topic_sentences[label].append(doc)

    return {"stay_id": num, "topics": topic_sentences}
