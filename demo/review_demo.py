from review.main_review import get_reviews, preprocess_reviews, use_model, get_review_conf
import argparse

def main():
    # argparse를 사용해 명령줄에서 URL 인자를 받을 수 있도록 설정
    parser = argparse.ArgumentParser(description="Airbnb 리뷰로부터 주제를 추출하는 데모입니다.")
    parser.add_argument(
        "--url",
        type=str,
        help="리뷰의 주제를 추출하고 싶은 Airbnb 숙소의 URL을 입력하세요.",
        default="https://www.airbnb.co.kr/rooms/44005242?check_in=2025-06-13&check_out=2025-06-15&photo_id=1265571749&source_impression_id=p3_1749786693_P3ao1iX7quls1JfM&previous_page_section_name=1000"
    )

    '''
    사용 예시:
    1. 특정 숙소 URL을 입력하여 실행:
    python demo/review_demo.py --url [Airbnb 숙소 URL]

    2. URL 인자를 생략하고 기본 URL로 실행:
    python demo/review_demo.py
    '''

    review_conf = get_review_conf()

    # 숙소 url
    url = parser.parse_args().url
    # 리뷰 크롤링
    data, num = get_reviews(url, review_conf['headers'])  # 리뷰 데이터와 숙소 번호 가져오기
    # 리뷰 전처리
    docs = preprocess_reviews(data, num)  # 전처리된 문장 리스트 생성
    # 모델 사용
    _ = use_model(docs, review_conf['seed_topics'])  # BERTopic 모델로 주제 분석 수행

if __name__ == '__main__':
    main()
