from requests import request, get
from requests.compat import urlparse, urlunparse, urljoin
from bs4 import BeautifulSoup
import re
import time
import base64
from urllib.parse import quote, unquote, urlencode
import json
import requests
import csv
from requests import request
import pandas as pd
from datetime import datetime



'''
c = cookie
'''

#
cookies = dict()
for line in c.splitlines():
    if len(line.split()) > 2:
        cookies[line.split()[0]]=line.split()[1]
print(cookies)

# type your heaeders
headers = {
            "Content-Type": "application/json",
            "User-Agent": "",
            "Accept": "*/*",
            "Referer": "",
            "Accept-Language": "",
            "api-key": "",
        }


# 도시 페이지 이동
def make_cursor(page: int) -> str:
    items_offset = (page - 1) * 18
    obj = {"section_offset": 0, "items_offset": items_offset, "version": 1}
    json_str = json.dumps(obj, separators=(',', ':'))
    b64 = base64.b64encode(json_str.encode()).decode()
    return quote(b64)

# 번역본 없으면 걍 원본 ㄱㄱ가 아니라 걍 없애
def comments(r):
    try:
        text = r["localizedReview"]["comments"]
    except:
        try:
            if r.get("language") == 'ko':
                text = r["comments"]
            else:
                return None
        except:
            return None

    # HTML 태그 정리 로직을 직접 여기에 포함
    text = re.sub(r'<br\s*/?>|<br></br>', '\n', text)  # <br> 계열 줄바꿈 처리
    text = re.sub(r'<.*?>', '', text)  # 그 외 HTML 태그 제거
    return text.strip()

# 얘는 뭐냐면 country 가져올라고 한것임 ㅋㅋ
def getCountry(stay_id, headers=headers):
    jurl = '' # json url
    variables = {
    "id":stay_id,
    "pdpSectionsRequest":{
        "adults":"1",
        "amenityFilters":None,
        "bypassTargetings":False,
        "categoryTag":"Tag:8536",
        "causeId":None,
        "children":"0",
        "disasterId":None,
        "discountedGuestFeeVersion":None,
        "displayExtensions":None,
        "federatedSearchId":"cd5c6974-f576-4c4f-9cb5-13d113ec51f9",
        "forceBoostPriorityMessageType":None,
        "hostPreview":False,
        "infants":"0",
        "interactionType":None,
        "layouts":["SIDEBAR","SINGLE_COLUMN"],
        "pets":0,
        "pdpTypeOverride":None,
        "photoId":"1011211469",
        "preview":False,
        "previousStateCheckIn":None,
        "previousStateCheckOut":None,
        "priceDropSource":None,
        "privateBooking":False,
        "promotionUuid":None,
        "relaxedAmenityIds":None,
        "searchId":None,
        "selectedCancellationPolicyId":None,
        "selectedRatePlanId":None,
        "splitStays":None,
        "staysBookingMigrationEnabled":False,
        "translateUgc":None,
        "useNewSectionWrapperApi":False,
        "sectionIds":None,
        "checkIn":"2025-06-22",
        "checkOut":"2025-06-27",
        "p3ImpressionId":"p3_1746161344_P31kUKCqN5sIEyE-"
        }
    }
    extensions = {
        "persistedQuery":{
            "version":1,
            "sha256Hash":"f813c4581945853f08f84b86d5de8e72c5015fe26be1e2ecfdd078a0e60eee7e"
        }
    }

    params = {
        "operationName":"StaysPdpSections",
        "locale":"ko",
        "currency":"KRW",
        "variables":json.dumps(variables, separators=(',',':')),
        "extensions":json.dumps(extensions, separators=(',',':')),
    }

    resp = requests.get(jurl+urlencode(params, quote_via=quote), headers=headers)
    rj = resp.json()
    breadcrumbDetails = rj["data"]["presentation"]["stayProductDetailPage"]["sections"]["metadata"]["seoFeatures"]["breadcrumbDetails"]
    country = breadcrumbDetails[0]["searchText"].split(', ')
    # 나라랑 시만 할거임
    if len(country) >= 2:
        city = country[-2]
        country = country[-1]
    elif len(country) == 1:
        city = None
        country = country[0]
    else:
        city = None
        country = None

    return country, city, breadcrumbDetails[1]["linkText"]

# 얘는 리뷰가져올 json임
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

# 유저 정보랑 그 유저가 어디로 갔는지
def getUserInfo(userid, headers=headers, cookies=cookies):
    jurl = '' # json url

    encoding = 'User:'+str(userid)
    userId = base64.b64encode(encoding.encode('utf-8')).decode('utf-8')

    variables = {
        "userId":userId,
        "isPassportStampsEnabled":True,
        "mockIdentifier":None,
        "fetchCombinedSportsAndInterests":True
        }

    extensions = {
        "persistedQuery":{
            "version":1,
            "sha256Hash":"79956c47b34f6cc315fc45325be7d2faaa8931e37d02530e0fc42590e7395004"
            }
        }


    params = {
        'operationName':'GetUserProfile',
        'locale':'ko',
        'currency':'KRW',
        'variables':json.dumps(variables, separators=(',',':')),
        'extensions':json.dumps(extensions, separators=(',',':'))
    }

    resp = requests.get(jurl+urlencode(params, quote_via=quote), headers=headers, cookies=cookies)

    if resp.status_code != 429:
        r = resp.json()
        months, years = None, None
        # 나이
        try:
            prompts = r['data']['node']['prompts']['editProfilePrompts']['prompts']
            for p in prompts:
                if p['fieldId'] == 'BIRTH_DECADE':
                    two = re.search(r'\d{2}', p['toggleInputField']['binaryInput']['helpText']).group()
                    if two[0] == '0':
                        age = int('20'+two)
                    else:
                        age = int('19'+two)
                    break
        except:
            age = None
        # 여행 횟수
        try:
            tripCount = int(r['data']['node']['profileInfo']['pastTripsCount'])
        except:
            tripCount = None
        # 호스트한테 리뷰 답장온 횟수
        try:
            reviewsCount = int(r['data']['presentation']['userProfileContainer']['userProfile']['reviewsReceivedFromHosts']['count'])
        except:
            reviewsCount = None
        # 여행 장소
        try:
            tripPlace = r['data']['node']['profileInfo']['aggregatedPastLocations']['edges']
        except:
            tripPlace = None
        try:
            months = int(r['data']['presentation']['userProfileContainer']['userProfile']['timeAsUser']['months'])
        except:
            months = None
        try:
            years = int(r['data']['presentation']['userProfileContainer']['userProfile']['timeAsUser']['years'])
        except:
            years = None
        # 가입 기간
        if months or years:
            totalPeriod = round(years+round(months/12,1))
        else:
            totalPeriod = None

        with open(os.path.join(save_base_path, "user.csv"), "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([userid, age, reviewsCount, tripCount, totalPeriod])

        with open(os.path.join(save_base_path, "tripPlace.csv"), "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if tripCount:
                for t in tripPlace:
                    try:
                        tripCountry = t['node']['stamp']['localizedLocation']
                        country, city = tripCountry.split(' ', 1)
                        tripDate = t['node']['stamp']['subtitle']
                        if '월' in tripDate:
                            tripDate = datetime.strptime(tripDate, '%Y년 %m월').strftime('%Y-%m')
                            writer.writerow([userid, country, city, tripDate])
                    except:
                        pass

    else:
        with open(os.path.join(save_base_path, "dropUsers.csv"), "a", newline="", encoding="utf-8") as f:
              writer = csv.writer(f)
              writer.writerow([userid])


cities = ['New York'
        ,'Los Angeles'
        ,'San Francisco'
        ,'Las Vegas'
        ,'Miami'
        ,'Chicago'
        ,'Orlando'
        ,'Vancouver'
        ,'Toronto'
        ,'Montreal'
        ,'Quebec City'
        ,'London'
        ,'Edinburgh'
        ,'Manchester'
        ,'Liverpool'
        ,'Paris'
        ,'Nice'
        ,'Lyon'
        ,'Marseille'
        ,'Berlin'
        ,'Munich'
        ,'Hamburg'
        ,'Frankfurt'
        ,'Rome'
        ,'Venice'
        ,'Milan'
        ,'Florence'
        ,'Naples'
        ,'Barcelona'
        ,'Seville'
        ,'Madrid'
        ,'Valencia'
        ,'Sydney'
        ,'Melbourne'
        ,'Brisbane'
        ,'Perth'
        ,'Tokyo'
        ,'Osaka'
        ,'Kyoto'
        ,'Fukuoka'
        ,'Sapporo'
        ,'Seoul'
        ,'Busan'
        ,'Jeju'
        ,'Gangneung'
        ,'Beijing'
        ,'Shanghai'
        ,'Guangzhou'
        ,'Chengdu'
        ,'Bangkok'
        ,'Phuket'
        ,'Chiang Mai'
        ,'Pattaya'
        ,'Hanoi'
        ,'Ho Chi Minh'
        ,'Mexico City'
        ,'Da Nang'
        ,'Cancun'
        ,'Guadalajara'
        ,'Rio de Janeiro'
        ,'São Paulo'
        ,'Istanbul'
        ,'Cappadocia'
        ,'Antalya'
        ]

ncities = list()
for c in cities:
        if ' ' in c:
                ncities.append(c.split(' ')[0]+'-'+c.split(' ')[1])
        else:
                ncities.append(c)
cities = ncities

Curl = list()
for i in range(len(cities)):
    Curl.append(f'https://www.yourURL.com/s/{cities[i]}/homes') # url

# cpgs = list() # 뉴욕이면 뉴욕에 대한 페이지들
# while Curl:
#     curl = Curl.pop(0) # 첫번째 뉴욕 -> ...

#     for i in range(1): # 15페이지까지 있으니깐
#         cursor = make_cursor(i)
#         cpgs.append(f"{curl}?cursor={cursor}")


#
import os
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
save_base_path = '/content/drive/MyDrive/Dataset/reviews'


#
userDict = dict()
stayDict = dict()
df1 = pd.read_csv(os.path.join(save_base_path, "user.csv"), encoding="utf-8")
df2 = pd.read_csv(os.path.join(save_base_path, "stay.csv"), encoding="utf-8")
for i in df1["user_id"]:
  userDict[i]=1
for i in df2["stay_id"]:
    stayDict[i]=1
len(userDict), len(stayDict)


# !rm -rf /content/drive
# import os
# from google.colab import drive
# drive.mount('/content/drive', force_remount=True)

# save_base_path = '/content/drive/MyDrive/Dataset/reviews'  # 여기에 저장할 거야
# os.makedirs(save_base_path, exist_ok=True)

# with open(os.path.join(save_base_path, "reviews.csv"), "w", newline="", encoding="utf-8") as f:
#     writer = csv.writer(f)
#     writer.writerow(["user_id", "name", "stay_id", "comments", "rating", "createdAt"])

# with open(os.path.join(save_base_path, "stay.csv"), "w", newline="", encoding="utf-8") as f:
#     writer = csv.writer(f)
#     writer.writerow(["stay_id", "stayType", "country", "city", "reviewCount"])

# with open(os.path.join(save_base_path, "user.csv"), "w", newline="", encoding="utf-8") as f:
#     writer = csv.writer(f)
#     writer.writerow(["user_id", "age", "reviewsCount", "tripCount", "registPeriod"])

# with open(os.path.join(save_base_path, "tripPlace.csv"), "w", newline="", encoding="utf-8") as f:
#     writer = csv.writer(f)
#     writer.writerow(["user_id", "country", "city","tripDate"])

# with open(os.path.join(save_base_path, "dropUsers.csv"), "w", newline="", encoding="utf-8") as f:
#     writer = csv.writer(f)
#     writer.writerow(['user_id'])

# !ls "/content/drive/MyDrive/Dataset/reviews"


#
while Curl:
    print(f"페이지수: {len(Curl)}")
    url = Curl.pop(0)
    URLs = list()
    resp = request('get',url)
    dom = BeautifulSoup(resp.text, 'html.parser')

    for a_tag in dom.select('a[href]'):
        if re.search(r'^/rooms/', a_tag['href']):
            link = urljoin('https://www.yourURL.com', a_tag['href'])
            if link not in URLs:
                URLs.append(link) # 나랑 똑같애야됨

    while URLs:
        url = URLs.pop(0)
        p1, p2 = re.split(r'[?]', url)
        review = p1+'/reviews?'+p2 # 리뷰 페이지 가는거

        url, num = re.split(r'/rooms/', p1) # num은 숙소번호
        num = num.split('/reviews')[0] # 진짜 숙소번호
        if num not in stayDict:
            stayDict[num]=1
            encoding = 'StayListing:'+num # 인코딩 하려고
            stay_id = base64.b64encode(encoding.encode('utf-8')).decode('utf-8')

            country, city, stayType = getCountry(stay_id, headers=headers)
            limit = 100
            offset = 0
            while 1:
                jresp = getReviewsJson(stay_id, limit, offset)
                try:
                    review_data = jresp["data"]["presentation"]["stayProductDetailPage"]["reviews"]
                    reviewsCount = review_data['metadata']['reviewsCount']
                    with open(os.path.join(save_base_path, "reviews.csv"), "a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        for r in review_data["reviews"]:
                            if comments(r):
                                userid = r["reviewer"]["id"]
                                name = r["reviewer"]["firstName"]
                                rating = r['rating']
                                if rating:
                                    rating = int(rating)
                                createdAt = r['createdAt']
                                if createdAt:
                                    createdAt = datetime.strptime(createdAt,"%Y-%m-%dT%H:%M:%SZ").date()
                                writer.writerow([
                                    int(userid), name, int(num), comments(r), rating, createdAt # "user_id", "name", "stay_id", "comments", "rating", "createdAt"
                            ])
                                if userid not in userDict:
                                    userDict[userid]=1
                                    getUserInfo(userid)

                except KeyError:
                    print("끝났거나 에러 발생")
                    break

                if offset >= int(reviewsCount) or limit >= int(reviewsCount):
                    break

                offset += limit

            with open(os.path.join(save_base_path, "stay.csv"), "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([num, stayType, country, city, reviewsCount]) # "stay_id", "stayType", "country", "city", "reviewCount"
