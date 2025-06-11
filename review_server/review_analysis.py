#review_analysis.py로 저장
import base64, re, requests, json
import pandas as pd
from bs4 import BeautifulSoup
from collections import defaultdict

# 모델 및 분석 도구 관련 라이브러리
import torch
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
import hdbscan
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import kss

from urllib.parse import quote, urlencode

# 리뷰 내용에서 한글 또는 번역본 코멘트를 추출
def comments(r):
    if r["language"] == 'ko':
        return r["comments"]
    else:
        try:
            return r["localizedReview"]["comments"]
        except:
            return None

# Airbnb 리뷰 API 요청 시 필요한 헤더
headers = {
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0",
    "Accept": "*/*",
    "Referer": "https://www.airbnb.co.kr/",
    "Accept-Language": "ko-KR,ko;q=0.9",
    "x-airbnb-api-key": "d306zoyjsyarp7ifhu67rjxn52tv0t20",
}

def getReviewsJson(stay_id, limit, offset, headers=headers):
    jurl = "https://www.airbnb.co.kr/api/v3/StaysPdpReviewsQuery/dec1c8061483e78373602047450322fd474e79ba9afa8d3dbbc27f504030f91d?"
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

# 리뷰 텍스트 전처리 함수
def clean_text(text):
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'<br.*?>', ' ', text)  # br 태그 제거
    text = re.sub(r'[^\w\s.,!?가-힣]', '', text)  # 특수문자 제거
    text = re.sub(r'([.!?])', r'\1 ', text)  # 구두점 뒤에 공백
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# 숙소 URL 기반으로 리뷰 분석 수행 함수
def run_topic_model_on_room(room_url):
    # 1. 숙소 ID 추출 및 stay_id 인코딩
    p1, _ = room_url.split("?")
    _, num = p1.split("/rooms/")
    num = num.split("/")[0]
    encoding = 'StayListing:' + num
    stay_id = base64.b64encode(encoding.encode('utf-8')).decode('utf-8')

    # 2. 리뷰 수집
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

    # 3. 리뷰 DataFrame 생성 및 검증
    reviews = pd.DataFrame(data).dropna()
    if len(reviews) < 5:
        raise ValueError("리뷰가 충분하지 않습니다.")

    # 4. 리뷰 문장 단위로 분해
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
                rows.append({"stay_id": num, "user_id": row['user_id'], "splitNum": split_num, "sentence": s, "start": start, "end": end})
                split_num += 1

    # 5. 고유 문장 리스트 준비
    sentence_df = pd.DataFrame(rows).dropna()
    docs = sentence_df['sentence'].drop_duplicates().tolist()
    if len(docs) < 5:
        raise ValueError("문장 수가 너무 적습니다.")

    # 6. 임베딩 모델 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embedding_model = SentenceTransformer("jhgan/ko-sbert-nli", device=device)

    # 7. 사전 정의된 토픽 키워드 (seed_topics)와 이름
    seed_topics = [
    ["청결도", "깨끗함", "더럽", "청소", "위생", "깔끔", "냄새"],
    ["위치", "가까움", "교통", "중앙", "접근성"],
    ["가격", "가성비", "비쌈", "저렴함", "요금"],
    ["시설", "편리함", "기능", "낡음", "새거", "장비"],
    ["호스트", "응대", "서비스", "친절함", "체크인", "체크아웃"],
    ["소음", "시끄러움", "조용함", "방음", "소란"]
    ]
    topic_names = ["청결도", "위치", "가격", "시설", "호스트", "소음"]

    # 8. 차원 축소 및 클러스터링 모델 정의
    umap_model = UMAP(n_components=2, random_state=42, metric='cosine', n_neighbors=30, min_dist=0.1)
    hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=6, min_samples=3, cluster_selection_method='leaf')

    # 9. BERTopic 모델 구성 및 학습
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

    # 10. seed topic과 cosine similarity로 실제 토픽 이름 매핑
    topic_embeddings = topic_model.topic_embeddings_[1:]
    seed_embeddings = embedding_model.encode([" ".join(keywords) for keywords in seed_topics])
    similarity = cosine_similarity(topic_embeddings, seed_embeddings)
    mapped_topics = {topic_ids[i]: topic_names[sim_row.argmax()] for i, sim_row in enumerate(similarity)}

    # 11. 문장들을 주제별로 분류
    topic_sentences = defaultdict(list)
    for doc, topic_id in zip(docs, topics):
        label = mapped_topics.get(topic_id, "기타")
        if label != "기타":
            topic_sentences[label].append(doc)

    # 12. 최종 결과 반환
    return {"stay_id": num, "topics": topic_sentences}
