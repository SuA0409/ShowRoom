document.addEventListener('DOMContentLoaded', () => {
  // 공통: 효과음
  const clickSound = document.getElementById('click-sound');

  // 서버 주소
  const SERVER_BASE_URL = 'https://4f8d-34-87-1-83.ngrok-free.app';

  // 닫기 버튼
  const closeBtn = document.getElementById('close-popup');
  const loadingContainer = document.getElementById('loading-container');
  if (closeBtn) {
    closeBtn.addEventListener('click', () => window.close());
  }

  // (옵션) 로고 버튼
  const logoBtn = document.getElementById('logo-button');
  if (logoBtn) {
    logoBtn.addEventListener('click', () => {
      clickSound.currentTime = 0;
      clickSound.play();
      window.open('https://your-domain/showroom', '_blank');
    });
  }

  // 3D 변환 관련 버튼
  const convertBtn = document.getElementById('convert');
  // 2D 생성 버튼
  const create2DBtn = document.getElementById('create-2d');
  // 상태 표시 엘리먼트
  const status = document.getElementById('status');
  // 리뷰 요약 버튼
  const analyzeBtn = document.getElementById('analyze-review');
  // 공통 버튼: 토글, 댓글 이동, 뒤로가기
  const toggleBtn = document.getElementById('toggle');
  const gotoBtn = document.getElementById('goto-comments');
  const backBtn = document.getElementById('back');

  // 썸네일 렌더링 함수
  function renderThumbnails(images) {
    const container = document.getElementById('selected-thumbnails');
    const countEl = document.getElementById('selected-count');
    if (!container) return;

    container.innerHTML = '';
    if (countEl) countEl.textContent = `선택된 이미지: ${images.length}장`;

    images.forEach(src => {
      const img = document.createElement('img');
      img.src = src;
      img.addEventListener('click', () => {
        chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
          chrome.tabs.sendMessage(tabs[0].id, { action: 'toggleImage', src }, (newImages) => {
            renderThumbnails(newImages);
          });
        });
      });
      container.appendChild(img);
    });
  }

  // 3D 변환 버튼 초기화 및 페이지 로직
  if (convertBtn) {
    const wrapper = convertBtn.parentElement;
    wrapper.addEventListener('click', () => {
      clickSound.currentTime = 0;
      clickSound.play();
    });

    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      const tab = tabs[0];
      if (!tab.url.includes('airbnb.co.kr') && !tab.url.includes('airbnb.com')) {
        alert('Airbnb 상세 페이지에서 실행해주세요.');
        return;
      }

      chrome.tabs.sendMessage(tab.id, { action: 'getImages' }, (response) => {
        if (chrome.runtime.lastError) {
          console.error('메시지 전송 실패:', chrome.runtime.lastError.message);
          alert('Airbnb 페이지에서 이미지를 찾을 수 없습니다.');
          return;
        }
        if (!Array.isArray(response)) {
          alert('이미지 데이터를 가져오지 못했습니다.');
          return;
        }

        renderThumbnails(response);
        convertBtn.disabled = response.length < 3;
        setupConvertButton(tab.id);
      });
    });
  }

  // 3D 변환 요청 셋업 함수
  function setupConvertButton(tabId) {
    convertBtn.replaceWith(convertBtn.cloneNode(true));
    const newBtn = document.getElementById('convert');
    newBtn.disabled = false;

    newBtn.addEventListener('click', () => {
      clickSound.currentTime = 0;
      clickSound.play();

      if (loadingContainer) loadingContainer.style.display = 'block';

      chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        chrome.tabs.sendMessage(tabs[0].id, { action: 'getImages' }, async (images) => {
          if (chrome.runtime.lastError) {
            console.error('선택된 이미지 요청 실패:', chrome.runtime.lastError.message);
            alert('선택된 이미지를 다시 가져올 수 없습니다.');
            return;
          }
          if (!Array.isArray(images) || images.length < 3) {
            alert('최소 3장의 이미지를 선택해야 합니다.');
            return;
          }

          renderThumbnails(images);

          try {
            const res = await fetch(`${SERVER_BASE_URL}/3d_upload`, {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
                'ngrok-skip-browser-warning': 'true'
              },
              body: JSON.stringify({ images })
            });
            const data = await res.json();

            if (data.status === 'success') {
              showStatus('✅ Fast3R 처리 완료 응답 받음!', true);
              console.log('✅ Fast3R 응답:', data);

              // 2D 버튼 활성화
              if (create2DBtn) {
                create2DBtn.disabled = false;
                create2DBtn.setAttribute('data-tooltip', '선택된 이미지 2D 재생성');
              }
            } else {
              showStatus('❌ Fast3R 처리 중 오류가 발생했습니다.', false);
            }
          } catch (err) {
            console.error('Fast3R 요청 중 오류:', err);
            showStatus('❌ 서버 통신 오류가 발생했습니다.', false);
          } finally {
            if (loadingContainer) loadingContainer.style.display = 'none';
          }
        });
      });
    });
  }

  // 2D 생성 버튼 클릭 이벤트
  if (create2DBtn) {
    create2DBtn.addEventListener('click', async () => {
      clickSound.currentTime = 0;
      clickSound.play();

      if (loadingContainer) loadingContainer.style.display = 'block';

      create2DBtn.disabled = true;
      create2DBtn.innerHTML  = '2D<br>생성 중...';

      try {
        const res = await fetch(`${SERVER_BASE_URL}/2d_upload`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'ngrok-skip-browser-warning': 'true'
          },
          body: JSON.stringify({})
        });
        const data = await res.json();

        if (data.status === 'success') {
          showStatus('✅ 2D 생성 완료!', true);
          console.log('✅ 2D 생성 응답:', data);
        } else {
          showStatus('❌ 2D 생성 중 오류 발생.', false);
        }
      } catch (err) {
        console.error('2D 생성 요청 중 오류:', err);
        showStatus('❌ 2D 생성 중 서버 오류.', false);
      } finally {
        if (loadingContainer) loadingContainer.style.display = 'none';
        create2DBtn.disabled = false;
        create2DBtn.textContent = '2D 생성';
      }
    });
  }

  // 상태 표시 함수
  function showStatus(message, isSuccess = true) {
    const statusEl = document.getElementById('status');
    statusEl.textContent = message;
    statusEl.className = isSuccess ? 'success' : 'error';
    statusEl.style.display = 'block';
    setTimeout(() => { statusEl.style.display = 'none'; }, 3000);
  }

  // 리뷰 요약 로직
  if (analyzeBtn) {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      if (tabs[0].url.includes('airbnb.co.kr/rooms/')) analyzeBtn.disabled = false;
    });

    analyzeBtn.addEventListener('click', () => {
      clickSound.currentTime = 0;
      clickSound.play();

      analyzeBtn.disabled = true;
      analyzeBtn.textContent = '분석 중...';

      chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        const url = tabs[0].url;
        if (!url.includes('airbnb.co.kr/rooms/')) {
          alert('Airbnb 숙소 상세 페이지에서만 사용할 수 있습니다.');
          analyzeBtn.disabled = false;
          analyzeBtn.textContent = '댓글 요약 시작';
          return;
        }

        fetch(`${SERVER_BASE_URL}/analyze_review`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ url })
        })
          .then(res => res.json())
          .then(data => {
            if (data.view_url) window.open(data.view_url, '_blank');
            else showStatus('❌ 분석 실패: ' + (data.error || 'Unknown error'), false);
          })
          .catch(e => showStatus('❌ 서버 오류: ' + e, false))
          .finally(() => {
            analyzeBtn.disabled = false;
            analyzeBtn.textContent = '댓글 요약 시작';
          });
      });
    });
  }

  // 기능 토글 버튼
  if (toggleBtn) {
    chrome.storage.local.get('enabled', data => {
      toggleBtn.textContent = data.enabled ? 'off' : 'on';
    });

    toggleBtn.addEventListener('click', () => {
      chrome.storage.local.get('enabled', data => {
        const newState = !data.enabled;
        chrome.storage.local.set({ enabled: newState }, () => {
          toggleBtn.textContent = newState ? 'off' : 'on';
          chrome.tabs.query({ active: true, currentWindow: true }, tabs => {
            chrome.tabs.sendMessage(tabs[0].id, { action: 'toggle', enabled: newState });
          });
        });
      });
    });
  }

  // 페이지 이동 버튼
  if (gotoBtn) gotoBtn.addEventListener('click', () => window.location.href = 'popup2.html');
  if (backBtn) backBtn.addEventListener('click', () => window.location.href = 'popup.html');
});