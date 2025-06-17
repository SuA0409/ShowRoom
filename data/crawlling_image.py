START, END = 0, 10 # 이미지 리스트 중 크롤링한 숙소들
save_base_path = '/content/drive/MyDrive/Dataset/city1'  # 이미지 저장 위치

SLEEP = 2
ROOM_LIST = ['주방', '거실', '침실'] # 실내 공간만 수집하기 위한 제한 사항

# Selenium으로 웹사이트를 크롤링하기 위한 준비 작업과 드라이버 설정

os.makedirs(save_base_path, exist_ok=True)
sys.path.insert(0, '/usr/lib/chromium-browser/chromedriver')

# 셀레니움 드라이버 설정 (Colab용)
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument('--window-size=1920x1080')
driver = webdriver.Chrome(options=chrome_options)

# Import
import os, re, json
from bs4 import BeautifulSoup
import requests
from requests import request
import base64
from urllib.parse import quote, urljoin
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import sys
import json

# 사전에 정의한 도시들의 리스트
with open('configs/city_list.json', 'r', encoding='utf-8') as f:
    cities = json.load(f)

cities = [it.replace(' ', '-') for it in cities["city1"]]
Curl = list()
for i in range(len(cities)):
    Curl.append(f'https://www.yourURL.com/{cities[i]}/homes')

def save_images(img_list, category, room_dir):
    '''크롤링한 이미지를 저장'''
    category_dir = os.path.join(room_dir, category)
    os.makedirs(category_dir, exist_ok=True)
    for idx_img, img_url in enumerate(img_list):
        try:
            img_data = requests.get(img_url, headers=headers).content
            img_filename = f"{category}_{idx_img}.jpg"
            img_path = os.path.join(category_dir, img_filename)
            with open(img_path, 'wb') as f:
                f.write(img_data)
            print(f" 저장 완료: {img_path}")
        except Exception as e:
            print(f" 이미지 다운로드 실패: {e}")

def make_cursor(page: int) -> str:
    '''다음 리스트로 도시를 옮기는 로직'''
    items_offset = (page - 1) * 18
    obj = {"section_offset": 0, "items_offset": items_offset, "version": 1}
    json_str = json.dumps(obj, separators=(',', ':'))
    b64 = base64.b64encode(json_str.encode()).decode()
    return quote(b64)

def crawl_and_save_images(driver, urls, i):
    '''숙소 url을 돌면서 방 이미지를 저장하는 로직'''
    idx = 0
    while urls:
        url = urls.pop()
        try:
            print(f"▶{i}/{END-1}▶ {idx+1}/{270}: {url}")

            # modal url 생성
            if '?' in url:
                p1, p2 = re.split(r'\?', url)
                modal_url = p1 + '?modal=PHOTO_TOUR_SCROLLABLE&' + p2
            else:
                modal_url = url + '?modal=PHOTO_TOUR_SCROLLABLE'

            driver.get(modal_url)
            time.sleep(SLEEP)

            soup = BeautifulSoup(driver.page_source, 'html.parser')

            # 이미지 분류할 리스트
            rooms = dict()

            for img_tag in soup.find_all('img', alt=True):
                if 12 > len(img_tag['alt']) > 6 and any(rl in img_tag['alt'] for rl in ROOM_LIST):
                    rooms[img_tag['alt'][:-5]] = rooms.get(img_tag['alt'][:-5], set())
                    rooms[img_tag['alt'][:-5]].add(img_tag['src'])

            # 이미지 가치 판단
            for category, img_set in rooms.items():
                if len(img_set) >= 3:
                    # 숙소 고유번호 추출
                    room_number = url.split('/rooms/')[-1].split('?')[0].strip()
                    if room_number in pre_room_list:
                        print(f'xxx 숙소 크롤링한 숙소:{room_number}')
                        continue
                    room_dir = os.path.join(save_base_path, room_number)
                    os.makedirs(room_dir, exist_ok=True)

                    with open(f'/content/drive/MyDrive/Dataset/room_number.txt', 'a', encoding='utf-8') as file:
                        file.write(f'{room_number}\n')

                    save_images(list(img_set), category, room_dir)

        except Exception as e:
            print(f"xxx 숙소 처리 중 오류: {e}")
            continue

        idx += 1

# Main
if __name__ == '__main__':
    urls = list()

    import time
    # --- 5. 도시별 숙소 링크 수집 함수 ---
    for i in range(START, END):
        current_time = time.time()


        pre_room_list = list()
        pre_room_list_path = '/content/drive/MyDrive/Dataset/room_number.txt'
        with open(pre_room_list_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]

        pre_room_list.extend(lines)
        pre_room_list = set(pre_room_list)

        curl = Curl[i]

        cpgs = list()
        for j in range(1, 16): # 15페이지까지 있으니깐
            cursor = make_cursor(j)
            cpgs.append(f"{curl}?cursor={cursor}")

        for urll in cpgs:
            resp = request('get',urll)
            dom = BeautifulSoup(resp.text, 'html.parser')

            for a_tag in dom.select('a[href]'):
                if re.search(r'^/rooms/', a_tag['href']):
                    link = urljoin('https://www.yourURL.com', a_tag['href'])
                    if link not in urls:
                        urls.append(link)

        crawl_and_save_images(driver, urls, i)
        print(f"\n 모든 작업 완료!\t소요 시간 : {time.time() - current_time}")