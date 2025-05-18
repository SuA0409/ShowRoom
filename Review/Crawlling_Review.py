# Import
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


#
c = '''
_aaj	22%7C1%7CUPOR1WjK%2Bo4nqFNgPemzOsBKhreydVN0y2HvDe7%2FP4MoGDrUMjOAoRs3PA3TjboOldnI5mJ4xDuhkb5r1l77%2B7NKhenEupQCeHpY0PDSP4bf9q%2BN8l8d96A0KrarZqZjGjLUvYy3RY5Gd%2BGfP%2BkrNYo0uHJVxxZJB28sGxR1I5Jk4DjEh76DYqycqcglNuHsrrNZyxJr%2BCRtsT6KMazySLUsWyHEAxDqqOGhMzbCCybJdizZ1iWsofNolq6wwBOZHdAlqygWDx94%2Bvf3G0sHrFUCaPgreOhGsBW6LawGIdbqkSUtHiC4QQofLJWhNzG48gipSdm11uUGx4zjWZhHqN5h1vdL%2Fe39eePXY4Nv6%2Bpr0Fk2OzyOFl6F15W1hrSqBoUXqmum%2F3wj2JqpB6IZm8khZmasoHV3o9qo4B2XzqjLeka01UmxHAumgW7yRpYKR9T%2F2BrxEegZ%2FcBIUVOZdL9uVLLCLdgBQoWBeRxPTqaBed1%2FBr8rkrKrOiVaja4%2FQA%2BnqbhM7RF2wvK5NxeRmW8Bws3P14ARrm0Mym%2BdsQujeV0hZ6TN7EblmEQ3taT5NIET%2FGoq4adNs%2BIzff5MVAXNapipn8%2FrY9CWJmuh%2F6gFfsx4sum1FHgZrQHLF5CFkch9Y4hUQfzDCERZi0I81MDA3ggTFqWNe8S3OswwWoNVKl1obnssfgcbO7tA%2BP1Gq5559EwwpVwHm3hlnRm7lyVmTd%2BXfGai5lQ7OZfBd4Vk%2FCp65QoXktRfKZtwLqQmrjk5MMv9Gh%2Bs%2BTgU%2FJzs3kBfnQuou2Ihdnz0aCU7lmLcTZLN7JPSqFk%3D	.airbnb.co.kr	/	2026-06-06T15:39:37.976Z	893	✓	✓				Medium
_aat	0%7CKyM1ZT%2BeuyhNwOHrP3u0XBnGN%2BiiIM4INNJMbypFMBfZcb9eOcs41iXcIPWGnRkA	.airbnb.co.kr	/	2026-06-06T15:39:37.976Z	76	✓	✓				Medium
_airbed_session_id	7b6d2071f6dbdfde2f1016b1ac479652	.airbnb.co.kr	/	2026-06-06T15:39:37.976Z	50	✓	✓				Medium
_cci	cban%3Aac-beb1d4c8-b81b-4e41-8621-2dab64913b7a	.airbnb.co.kr	/	2025-11-02T15:54:38.000Z	50		✓				Medium
_ccv	cban%3A0_183215%3D1%2C0_200000%3D1%2C0_183345%3D1%2C0_183243%3D1%2C0_183216%3D1%2C0_179751%3D1%2C0_200003%3D1%2C0_200005%3D1%2C0_179754%3D1%2C0_179750%3D1%2C0_179737%3D1%2C0_179744%3D1%2C0_179739%3D1%2C0_179743%3D1%2C0_179749%3D1%2C0_200012%3D1%2C0_200011%3D1%2C0_183217%3D1%2C0_183219%3D1%2C0_183096%3D1%2C0_179747%3D1%2C0_179740%3D1%2C0_179752%3D1%2C0_183241%3D1%2C0_200007%3D1%2C0_183346%3D1%2C0_183095%3D1%2C0_210000%3D1%2C0_210001%3D1%2C0_210002%3D1%2C0_210003%3D1%2C0_210004%3D1%2C0_210010%3D1%2C0_210012%3D1%2C0_210008%3D1%2C0_210016%3D1%2C0_210017%3D1	.airbnb.co.kr	/	2025-11-02T15:54:38.000Z	563		✓				Medium
_csrf_token	V4%24.airbnb.co.kr%24YQrCy45yGU4%24GyuIS_mzmF0CzAEC11rvUobs-zhQEXdf9Ld62TOMrEI%3D	.airbnb.co.kr	/	Session	92		✓				Medium
_ga	GA1.1.1517047731.1746200372	.airbnb.co.kr	/	2026-06-06T15:39:32.063Z	30						Medium
_ga_2P6Q8PGG16	GS1.1.1746200371.1.1.1746201280.0.0.497054865	.airbnb.co.kr	/	2026-06-06T15:54:40.355Z	59						Medium
_gcl_au	1.1.194414233.1746200433	.airbnb.co.kr	/	2025-07-31T15:40:32.000Z	31						Medium
_gtmeec	eyJlbSI6ImMzZTcxYzQ3ODNiZGZlMzcyODQ2ZGFmNWQ0NjliNmEzZmIwZTkyNjkzZTEwNDE2Zjc1YmE4YjVkODRjMzc1MTAiLCJwaCI6ImUzYjBjNDQyOThmYzFjMTQ5YWZiZjRjODk5NmZiOTI0MjdhZTQxZTQ2NDliOTM0Y2E0OTU5OTFiNzg1MmI4NTUiLCJsbiI6IjE1MDhiNjk3ODk1YWJmMDNkNTVjMzg0MWY1OTIzNmFiOTJjOWJhNmJhODk3OTVjODMzN2ZjZjM5MmZkZWU4YjQiLCJmbiI6ImViNjg3YWZiMGY0ODIzZThlYjgwYjNjMWMxZmE2NTE5Y2RjOTE2Y2Q4ZTMxYzYzMTA2ZDAzOWFjNWIwZmE5MDciLCJjdCI6ImUzYjBjNDQyOThmYzFjMTQ5YWZiZjRjODk5NmZiOTI0MjdhZTQxZTQ2NDliOTM0Y2E0OTU5OTFiNzg1MmI4NTUiLCJzdCI6ImUzYjBjNDQyOThmYzFjMTQ5YWZiZjRjODk5NmZiOTI0MjdhZTQxZTQ2NDliOTM0Y2E0OTU5OTFiNzg1MmI4NTUiLCJnZSI6ImUzYjBjNDQyOThmYzFjMTQ5YWZiZjRjODk5NmZiOTI0MjdhZTQxZTQ2NDliOTM0Y2E0OTU5OTFiNzg1MmI4NTUiLCJkYiI6IjkxODZiY2U4ODlkNWY1OWQ1OGJhY2YyNDNhNmU4OWRkNzkyYmM3ZDIzY2Q4OTQ1NjNjODNiODZkNzM2NjU1YzIiLCJjb3VudHJ5IjoiZTNiMGM0NDI5OGZjMWMxNDlhZmJmNGM4OTk2ZmI5MjQyN2FlNDFlNDY0OWI5MzRjYTQ5NTk5MWI3ODUyYjg1NSIsImV4dGVybmFsX2lkIjoiMTc0NjIwMDM2OV9FQVlXWXlaamsyT1dJd01tIn0%3D	.airbnb.co.kr	/	2025-07-31T15:54:40.868Z	941	✓	✓				Medium
_pt	1--WyJlMGRhNWI3ZGY4ZTg3MTBkYTY1ZmY0YjgwYzExNjExMWY2NTBiNTUxIl0%3D--73972e586a053f82afcfdcac3b70d7dd2e49f447	.airbnb.co.kr	/	2025-10-31T15:54:33.646Z	110		✓				Medium
_scid	5bbcec0c-fcba-46b8-6aea-c9973ccca15a	.airbnb.co.kr	/	2026-05-02T15:40:34.161Z	41		✓				Medium
_user_attributes	%7B%22curr%22%3A%22KRW%22%2C%22guest_exchange%22%3A1435.82557%2C%22id%22%3A692888664%2C%22id_str%22%3A%22692888664%22%2C%22hash_user_id%22%3A%22e0da5b7df8e8710da65ff4b80c116111f650b551%22%2C%22eid%22%3A%22eVzXArD_W74oCO-a9ciwng%3D%3D%22%2C%22num_h%22%3A0%2C%22name%22%3A%22Jun%22%2C%22num_action%22%3A0%2C%22is_admin%22%3Afalse%2C%22can_access_photography%22%3Afalse%2C%22travel_credit_status%22%3Anull%2C%22referrals_info%22%3A%7B%22receiver_max_savings%22%3Anull%2C%22receiver_savings_percent%22%3Anull%2C%22receiver_signup%22%3Anull%2C%22referrer_guest%22%3A%22%E2%82%A916%2C000%22%2C%22terms_and_conditions_link%22%3A%22%2Fhelp%2Farticle%2F2269%22%2C%22wechat_link%22%3Anull%2C%22offer_discount_type%22%3Anull%7D%7D	.airbnb.co.kr	/	2025-11-02T15:54:45.995Z	735		✓				Medium
ak_bmsc	BD6146D31BBA8DEEBC6B146D3E10F710~000000000000000000000000000000~YAAQJxQgFzBA+WaWAQAAwRinkRsWGHGAFLNcc3Y4eQLVVsNBFiETWKD0rKv/mf7cTKqs+VotqQUj8KUJgnrR5nictvIcIUsraLCWCPWxhFFpBNkFpGbRNfuhSkZCth4qUXYsuWm+fsOGVoH7z/GDCXaNr58UyNIL4d5SqJWz3gIP9QQFQIFDdu2BDyx4ukWFMzCUs0ag6OtZmORLXGSVjal/qmiJakvaVxKrlk+QWMWdLFGk6fRBYV20PncTIN/lz9mtEkllG6si9QAd1YPQk8ntpo3nR06uK8IFAMoTQ65fZ2jPZHwUhNxgeB9ay+O2OdGVlKL/iayF459bOiwSt8uPY+LYNCxjmKPYhStg9mNSCqSNS2pZ0zE8nt99TMIbDuHK0BZkvVlmxFKj	.airbnb.co.kr	/	2025-05-02T17:39:28.925Z	471						Medium
auth_jitney_session_id	2140eaa7-8183-4c68-96b8-15f4be58566d	.www.airbnb.co.kr	/	2025-11-02T15:39:32.000Z	58		✓				Medium
bev	1746200369_EAYWYyZjk2OWIwMm	.airbnb.co.kr	/	2026-06-06T15:54:47.995Z	30		✓				Medium
bm_sv	F08BC6179A33EC4192741F4305122CD2~YAAQXOn7ywApX3uWAQAAER+1kRvkadMauEAnnuowTI/d0F5ZkHwKAzlpOrCnda4BiUktbtrEVUHKiGa25nluoj78TquGbg70Bp4hgqJj3UVdK2grqwjNeltDbYqNXA9zr9r6anaePsKEo+bY12BHSBNFxI2xKUHa7jTO0swFc5lejGtR7q15EzvGSGmBI66bqol2wIaK9lQKbThJu/IVO2nlvD4csaOOQWKDkiRcMqAO3SyO6XVM4SXURNNm0hlYYrpI~1	.airbnb.co.kr	/	2025-05-02T17:39:29.995Z	300		✓				Medium
cbkp	3	.airbnb.co.kr	/	2025-11-02T15:54:14.000Z	5		✓				Medium
cdn_exp_5d4847f3128303184	control	.airbnb.co.kr	/	2025-07-01T15:39:32.408Z	32						Medium
cfrmfctr	DESKTOP	.airbnb.co.kr	/	2025-11-02T15:39:32.000Z	15		✓				Medium
everest_cookie	1746200369.EAZDY0OTRjNDBkYWI2OW.LWSaHfGPQf0XV2YbV49vdP1mGP_VpDyFoxRCcyg09do	.airbnb.co.kr	/	2026-06-06T15:39:28.925Z	89		✓				Medium
flags	0	.airbnb.co.kr	/	Session	6		✓				Medium
FPAU	1.1.590926196.1746200372	.airbnb.co.kr	/	2025-07-31T15:39:31.181Z	28		✓				Medium
FPGSID	1.1746200433.1746201280.G-2P6Q8PGG16.CdTiPUzAHnywJW-HzCVpkw	.airbnb.co.kr	/	2025-05-02T16:10:33.116Z	65		✓				Medium
FPID	FPID2.3.INrxUaCNw%2B7DJCB71jOH%2FpJodTqkkIvIF8la47qgfoM%3D.1746200371	.airbnb.co.kr	/	2026-06-06T15:54:40.698Z	73	✓	✓				Medium
FPLC	E%2BSroOy%2Bg0JRGPYLv1%2F13eJrLgwPdf5usmgYAXaMRkUCMPNI84KVrzy0r%2FR8nQEHif2RE10aE4vmbP3D0YJr%2BayOdfJvOGNLsAHZ43Ae0NO1MBKNqhSe%2Fw%2FlsZF4eQ%3D%3D	.airbnb.co.kr	/	2025-05-03T11:39:32.187Z	150		✓				Medium
frmfctr	wide	.airbnb.co.kr	/	2025-11-02T15:39:32.000Z	11		✓				Medium
hli	1	.airbnb.co.kr	/	2026-06-06T15:39:37.976Z	4		✓				Medium
hli	1	.www.airbnb.co.kr	/	2025-11-02T15:39:38.000Z	4		✓				Medium
jitney_client_session_created_at	1746200372.073	.www.airbnb.co.kr	/	2025-11-02T15:39:32.000Z	46		✓				Medium
jitney_client_session_id	f93b9a86-9aca-46d4-adc2-89fff89db409	.www.airbnb.co.kr	/	2025-11-02T15:39:32.000Z	60		✓				Medium
jitney_client_session_updated_at	1746201283.075	.www.airbnb.co.kr	/	2025-11-02T15:54:44.000Z	46		✓				Medium
li	1	.airbnb.co.kr	/	2026-06-06T15:39:37.976Z	3						Medium
oauth_popup	%7B%22success%22%3Atrue%2C%22data%22%3A%7B%7D%2C%22close_window%22%3Atrue%7D	www.airbnb.co.kr	/	2025-11-02T15:39:36.977Z	87						Medium
previousTab	%7B%22id%22%3A%222a0898d3-4ae6-4d0c-939f-718865419226%22%7D	.www.airbnb.co.kr	/	2025-11-02T15:54:48.000Z	70		✓				Medium
rclmd	%7B%22692888664%22%3A%22google%22%7D	.airbnb.co.kr	/	2026-06-06T15:39:37.976Z	41		✓				Medium
rclu	%7B%22692888664%22%3A%22ajhKpcGIqGjMs511pNlPDCkIYJfbLzJfVPIrAnSaJGo%3D%22%7D	.airbnb.co.kr	/	2026-06-06T15:39:37.976Z	80		✓				Medium
roles	0	.airbnb.co.kr	/	Session	6		✓				Medium
tzo	540	.airbnb.co.kr	/	2025-11-02T15:54:48.000Z	6		✓				Medium
'''


#
cookies = dict()
for line in c.splitlines():
    if len(line.split()) > 2:
        cookies[line.split()[0]]=line.split()[1]
print(cookies)

headers = {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36",
            "Accept": "*/*",
            "Referer": "https://www.airbnb.co.kr/",
            "Accept-Language": "ko-KR,ko;q=0.9",
            "x-airbnb-api-key": "d306zoyjsyarp7ifhu67rjxn52tv0t20",
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
    jurl = 'https://www.airbnb.co.kr/api/v3/StaysPdpSections/f813c4581945853f08f84b86d5de8e72c5015fe26be1e2ecfdd078a0e60eee7e?'
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

# 유저 정보랑 그 유저가 어디로 갔는지
def getUserInfo(userid, headers=headers, cookies=cookies):
    jurl = 'https://www.airbnb.co.kr/api/v3/GetUserProfile/79956c47b34f6cc315fc45325be7d2faaa8931e37d02530e0fc42590e7395004?'

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
    Curl.append(f'https://www.airbnb.co.kr/s/{cities[i]}/homes')

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
save_base_path = '/content/drive/MyDrive/AirbnbDataset/airbnb_reviews'


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

# save_base_path = '/content/drive/MyDrive/AirbnbDataset/airbnb_reviews'  # 여기에 저장할 거야
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

# !ls "/content/drive/MyDrive/AirbnbDataset/airbnb_reviews"


#
while Curl:
    print(f"페이지수: {len(Curl)}")
    url = Curl.pop(0)
    URLs = list()
    resp = request('get',url)
    dom = BeautifulSoup(resp.text, 'html.parser')

    for a_tag in dom.select('a[href]'):
        if re.search(r'^/rooms/', a_tag['href']):
            link = urljoin('https://www.airbnb.co.kr', a_tag['href'])
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
print('끝!!!!!!!!!!!!!!')
