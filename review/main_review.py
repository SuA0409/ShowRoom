#------------------------- 모델 ---------------------------
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
from requests import request
from bs4 import BeautifulSoup
import re
import base64
from urllib.parse import quote, urlencode
import json
import requests


def comments(korean):
    """
    한국어로 번역한것이 없을 때 or 한국어 댓글이 아닐때 None 반환 함수

    Args:
        korean (dict): 리뷰 데이터를 포함한 딕셔너리

    Returns:
        str or None: 한국어 리뷰 텍스트 또는 번역된 리뷰 텍스트 or 조건에 맞지 않으면 None
    """
    if korean["language"] == 'ko':  # 리뷰 언어가 한국어인지 확인
        return korean["comments"]  # 한국어 리뷰 텍스트 반환
    else:
        try:
            return korean["localizedReview"]["comments"]  # 번역된 리뷰 텍스트 반환
        except:
            return None  # 번역된 리뷰가 없으면 None 반환

def getReviewsJson(stay_id, limit, offset, headers):
    """
    리뷰에 해당하는 json 호출 함수

    Args:
        stay_id (str): Base64로 인코딩된 숙소 ID
        limit (int): 한 번에 가져올 리뷰 수
        offset (int): 리뷰 조회 시작 지점
        headers (dict): API 요청에 필요한 헤더 정보

    Returns:
        dict: 리뷰 데이터가 포함된 JSON 응답
    """
    jurl = "https://www.airbnb.co.kr/api/v3/StaysPdpReviewsQuery/dec1c8061483e78373602047450322fd474e79ba9afa8d3dbbc27f504030f91d?"  # API 기본 URL
    variables = {  # API 요청 변수 설정
        "id": stay_id,  # 숙소 ID
        "pdpReviewsRequest": {
            "fieldSelector": "for_p3_translation_only",  # 번역 리뷰만 선택
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

    resp = requests.get(jurl + urlencode(params, quote_via=quote), headers=headers)  # API 요청 전송
    return resp.json()  # JSON 응답 반환

def clean_text(text):
    """
    정규표현식으로 불용어 전처리 함수

    Args:
        text (str): 전처리할 리뷰 텍스트

    Returns:
        str: 불용어가 제거된 텍스트
    """
    text = text.replace('\n', ' ').replace('\r', ' ')  # 개행 문자 공백으로 변환
    text = text.replace('<br>', '').replace('<br/>', '').replace('<br />', '')  # HTML 줄바꿈 태그 제거
    text = re.sub(r'[^\w\s.,$!?가-힣]|[/\\[<a-z>]:\[\]{}|]', '', text)  # 특수문자 제거
    text = re.sub(r'[!\"“”‘’?.]+', '. ', text)  # 구두점 단순화
    text = re.sub(r'[():]', ' ', text)  # 괄호 제거
    text = re.sub(r'\s+', ' ', text).strip()  # 연속 공백 제거 및 양쪽 공백 제거
    return text.strip()  # 최종 전처리된 텍스트 반환

def get_reviews(url, headers):
    """
    리뷰에 해당하는 json 호출 함수

    Args:
        url (str): Airbnb 숙소 리뷰 페이지 URL
        headers (dict): API 요청에 필요한 헤더 정보

    Returns:
        tuple: (리뷰 데이터 리스트, 숙소 번호)
    """
    p1, p2 = re.split(r'[?]', url)  # URL에서 쿼리 문자열 분리
    url, num = re.split(r'/rooms/', p1)  # 숙소 번호 추출
    num = num.split('/reviews')[0]  # 순수 숙소 번호
    encoding = 'StayListing:' + num  # 숙소 ID 인코딩
    stay_id = base64.b64encode(encoding.encode('utf-8')).decode('utf-8')  # Base64 인코딩

    limit, cnt, offset = 50, 50, 0  # 한 번에 가져올 리뷰 수와 증가량 설정과 오프셋 초기화
    data = []  # 리뷰 데이터를 저장할 리스트
    while True:
        res_json = getReviewsJson(stay_id, limit, offset, headers)  # 리뷰 JSON 데이터 가져오기
        try:
            review_data = res_json["data"]["presentation"]["stayProductDetailPage"]["reviews"]  # 리뷰 데이터 접근
            reviewsCount = int(review_data['metadata']['reviewsCount'])  # 총 리뷰 수
            for r in review_data["reviews"]:  # 각 리뷰 처리
                comment_text = comments(r)  # 리뷰 텍스트 추출
                if comment_text:  # 유효한 리뷰 텍스트인 경우
                    data.append({  # 리뷰 데이터 추가
                        "user_id": r["reviewer"]["id"],  # 사용자 ID
                        "name": r["reviewer"]["firstName"],  # 사용자 이름
                        "stay_id": num,  # 숙소 번호
                        "comment": comment_text,  # 리뷰 텍스트
                        "rating": r["rating"],  # 평점
                        "createdAt": r["createdAt"]  # 작성일
                    })
        except KeyError:
            break  # 데이터 접근 오류 시 루프 종료

        if offset >= reviewsCount:  # 모든 리뷰를 가져왔으면 종료
            break
        offset += cnt  # 오프셋 증가
        limit += cnt  # 리뷰 수 증가

    return data, num  # 리뷰 데이터와 숙소 번호 반환

def preprocess_reviews(data, num):
    """
    리뷰 문장분리 및 불용어 처리하는 함수

    Args:
        data (list): 크롤링한 리뷰 데이터 리스트
        num (str): 숙소 번호

    Returns:
        list: 전처리된 문장 리스트

    Raises:
        ValueError: 리뷰 데이터가 5개 미만일 경우
    """
    reviews = pd.DataFrame(data).dropna().reset_index(drop=True)  # 리뷰 데이터를 DataFrame으로 변환 및 결측값 제거
    rows = []  # 전처리된 문장을 저장할 리스트
    for j, row in reviews.iterrows():  # 각 리뷰 처리
        raw_comment = row['comment']  # 원본 리뷰 텍스트
        user_id = row['user_id']  # 사용자 ID
        cleaned_comment = clean_text(raw_comment)  # 불용어 처리
        sentences = kss.split_sentences(cleaned_comment)  # KSS로 문장 분리
        idx = 0  # 텍스트 내 위치 추적
        split_num = 1  # 문장 번호 초기화

        for sentence in sentences:  # 각 문장 처리
            sentence = sentence.strip()  # 양쪽 공백 제거
            split_sentences = re.split(r'(?<=[가-힣\w])\.(?=[^\d])', sentence)  # 추가 문장 분리
            if not isinstance(split_sentences, list):  # 리스트가 아닌 경우 변환
                split_sentences = [sentence]
                
            for s in split_sentences:  # 각 세부 문장 처리
                s = s.strip()  # 공백 제거
                if len(s) < 7:  # 문장이 7자 미만이면 스킵
                    continue
                start = cleaned_comment.find(s, idx)  # 문장 시작 위치
                end = start + len(s) - 1  # 문장 끝 위치
                idx = end + 1  # 다음 위치로 이동
                rows.append({  # 전처리된 문장 정보 추가
                    "stay_id": num,  # 숙소 번호
                    "user_id": user_id,  # 사용자 ID
                    "splitNum": split_num,  # 문장 번호
                    "sentence": s,  # 문장 텍스트
                    "start": start,  # 시작 위치
                    "end": end  # 끝 위치
                })
                split_num += 1  # 문장 번호 증가

    sentence = pd.DataFrame(rows, columns=["stay_id", "user_id", "splitNum", "sentence", "start", "end"])  # DataFrame 생성
    sentence = sentence['sentence']  # 문장 열만 추출
    docs = sentence.dropna().drop_duplicates().tolist()  # NaN 및 중복 제거 후 리스트 변환
    if len(docs) < 5:  # 리뷰가 5개 미만이면 예외 발생
        raise ValueError("데이터가 충분하지 않습니다.")
    
    return docs  # 전처리된 문장 리스트 반환

def use_model(docs, seed_topics, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    """
    BERTopic 모델을 사용하여 리뷰 데이터를 주제별로 분석하는 함수

    Args:
        docs (list): 전처리된 리뷰 문장 리스트
        device (torch.device): 모델 실행 장치 (기본: GPU 또는 CPU)

    Returns:
        None: 주제별 리뷰를 콘솔에 출력
    """
    embedding_model = SentenceTransformer("jhgan/ko-sbert-nli", device=device)  # 한국어 Sentence-BERT 모델 로드
    topic_names = ["청결도", "소음", "위치", "가격", "시설", "호스트"]  # 주제 이름 정의

    umap_model = UMAP(n_components=2, random_state=42, metric='cosine', n_neighbors=30, min_dist=0.1)  # UMAP 설정
    hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=6, min_samples=3, cluster_selection_method='leaf')  # HDBSCAN 설정
    topic_model = BERTopic(  # BERTopic 모델 설정
        embedding_model=embedding_model,  # 임베딩 모델
        language="multilingual",  # 다국어 지원
        min_topic_size=6,  # 최소 토픽 크기
        nr_topics="auto",  # 토픽 수 자동 결정
        seed_topic_list=seed_topics,  # 시드 토픽 리스트
        umap_model=umap_model,  # UMAP 모델
        hdbscan_model=hdbscan_model,  # HDBSCAN 모델
        verbose=True,  # 진행 상황 출력
        vectorizer_model=TfidfVectorizer()  # TF-IDF 벡터화
    )

    topics, probs = topic_model.fit_transform(docs)  # BERTopic 모델로 토픽 추출
    topic_info = topic_model.get_topic_info()  # 토픽 정보 가져오기
    topic_ids = topic_info[topic_info['Topic'] != -1]['Topic'].tolist()  # 노이즈(-1) 제외한 토픽 ID 추출

    topic_embeddings = topic_model.topic_embeddings_[1:]  # 노이즈 제외한 토픽 임베딩
    seed_embeddings = embedding_model.encode([" ".join(keywords) for keywords in seed_topics])  # 시드 토픽 임베딩
    similarity = cosine_similarity(topic_embeddings, seed_embeddings)  # 토픽과 시드 간 유사도 계산

    mapped_topics = {}  # 토픽 매핑 딕셔너리
    for i, sim_row in enumerate(similarity):  # 각 토픽에 대해
        best_idx = sim_row.argmax()  # 가장 유사한 시드 토픽 인덱스
        mapped_topics[topic_ids[i]] = topic_names[best_idx]  # 토픽 이름 매핑

    topic_sentences = defaultdict(list)  # 주제별 문장 저장
    for doc, topic_id in zip(docs, topics):  # 각 문서와 토픽 ID 처리
        label = mapped_topics.get(topic_id, "기타")  # 토픽 레이블 가져오기
        if label != "기타":  # 기타 제외
            topic_sentences[label].append(doc)  # 주제별 문장 추가

    return topic_sentences