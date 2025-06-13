document.addEventListener('DOMContentLoaded', () => {
  // 서버 기본 URL 설정
  const SERVER_BASE_URL = 'add_your_main_ngrok_server_address!!';

  // 공통: 버튼 클릭 효과음 엘리먼트
  const clickSound = document.getElementById('click-sound');

   // UI 엘리먼트 정의: 닫기, 로딩 컨테이너
  const closeBtn = document.getElementById('close-popup');
  const loadingContainer = document.getElementById
  ('loading-container');

  // 닫기 버튼 클릭 시 팝업 닫기
  if (closeBtn) {
    closeBtn.addEventListener('click', () => window.close());
  }

  // (옵션) 로고 버튼 클릭 시 사운드 재생 및 새 탭으로 쇼룸 열기
  const logoBtn = document.getElementById('logo-button');
  if (logoBtn) {
    logoBtn.addEventListener('click', () => {
      clickSound.currentTime = 0;
      clickSound.play();
      window.open('https://your-domain/showroom', '_blank');
    });
  }

  // UI 엘리먼트 정의: 3D 변환, 2D 생성, 상태 표시, 리뷰 요약, 토글, 페이지 이동 버튼
  const convertBtn = document.getElementById('convert');
  const create2DBtn = document.getElementById('create-2d');
  const status = document.getElementById('status');
  const analyzeBtn = document.getElementById('analyze-review');
  const toggleBtn = document.getElementById('toggle');
  const gotoBtn = document.getElementById('goto-comments');
  const backBtn = document.getElementById('back');

  // 팝업 로드 시 저장된 이미지 목록 복원 및 썸네일 렌더링
  chrome.storage.local.get({ selectedImages: [] }, ({ selectedImages }) => {  // ★ ADDED
    if (selectedImages.length > 0) {                                          // ★ ADDED
      renderThumbnails(selectedImages);                                        // ★ ADDED
      convertBtn.disabled = selectedImages.length < 3;                         // ★ ADDED
      if (selectedImages.length >= 3) {                                        // ★ ADDED
        create2DBtn.disabled = false;                                          // ★ ADDED
        create2DBtn.setAttribute('data-tooltip', '선택된 이미지 2D 재생성');     // ★ ADDED
      }                                                                   // ★ ADDED
    }                                                                          // ★ ADDED
  });                                                                          // ★ ADDED

  // 썸네일 렌더링 및 클릭 이벤트 처리 함수
  function renderThumbnails(images) {
    const container = document.getElementById('selected-thumbnails');
    const countEl = document.getElementById('selected-count');
    if (!container) return;

    container.innerHTML = '';
    if (countEl) countEl.textContent = `선택된 이미지: ${images.length}장`;

    images.forEach(src => {
      const img = document.createElement('img');
      img.src = src;

      // 썸네일 클릭 시 이미지 선택/해제 및 UI 업데이트
      img.addEventListener('click', () => {
        chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
          chrome.tabs.sendMessage(tabs[0].id, { action: 'toggleImage', src }, (newImages) => {
            renderThumbnails(newImages);
            chrome.storage.local.set({ selectedImages: newImages });        // ★ ADDED
            convertBtn.disabled = newImages.length < 3;                     // ★ ADDED
            if (newImages.length >= 3) {                                    // ★ ADDED
              create2DBtn.disabled = false;                                 // ★ ADDED
              create2DBtn.setAttribute('data-tooltip', '선택된 이미지 2D 재생성'); // ★ ADDED
            }                                                              // ★ ADDED
          });
        });
      });
      container.appendChild(img);
    });
  }

  // 3D 변환 버튼 초기화 및 Airbnb 페이지 이미지 로드
  if (convertBtn) {
    const wrapper = convertBtn.parentElement;
    wrapper.addEventListener('click', () => {
      clickSound.currentTime = 0;
      clickSound.play();
    });

    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      const tab = tabs[0];
      // Airbnb 페이지 확인 및 이미지 가져오기
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

        // 최초 이미지 로드 직후 저장
        chrome.storage.local.set({ selectedImages: response });        // ★ ADDED
      });
    });
  }

  // 3D 변환 요청 설정 및 처리 함수
  function setupConvertButton(tabId) {
    const oldBtn = document.getElementById('convert');
    const newBtn = oldBtn.cloneNode(true); // 버튼 복제
    oldBtn.parentNode.replaceChild(newBtn, oldBtn); // 기존 버튼 교체
    newBtn.disabled = false;
  
    let isRunning = false; // 중복 실행 방지용 플래그
  
    newBtn.addEventListener('click', async () => {
      if (isRunning) return; // 중복 방지
      isRunning = true;
  
      clickSound.currentTime = 0;
      clickSound.play();
  
      if (loadingContainer) loadingContainer.style.display = 'block';
  
      try {
        const tabs = await new Promise(resolve =>
          chrome.tabs.query({ active: true, currentWindow: true }, resolve)
        );
  
        const images = await new Promise((resolve, reject) => {
          chrome.tabs.sendMessage(tabs[0].id, { action: 'getImages' }, (res) => {
            if (chrome.runtime.lastError) return reject(chrome.runtime.lastError);
            resolve(res);
          });
        });
  
        if (!Array.isArray(images) || images.length < 3) {
          alert('최소 3장의 이미지를 선택해야 합니다.');
          return;
        }
  
        renderThumbnails(images);
  
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
  
          // viser 새 창 열기
          if (data.viser_response && data.viser_response.status) {
            const match = data.viser_response.status.match(/"(https:\/\/[^\s"]+)"/);
            if (match && match[1]) {
              window.open(match[1], '_blank');
            }
          }
  
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
        isRunning = false;
      }
    });
  }


  // 2D 생성 버튼 클릭 이벤트 처리
  if (create2DBtn) {
    create2DBtn.addEventListener('click', async () => {
      clickSound.currentTime = 0;
      clickSound.play();

      if (loadingContainer) loadingContainer.style.display = 'block';

      create2DBtn.disabled = true;
      create2DBtn.innerHTML  = '2D<br>생성 중...';

      try {
        // 서버에 2D 생성 요청 전송
        const res = await fetch(`${SERVER_BASE_URL}/2d_upload`, {
          method: 'POST',
          headers: {
            'ngrok-skip-browser-warning': 'true'
          },
        });
        const data = await res.json();

        // 서버 응답 처리
        if (data.status === 'success') {
          showStatus('✅ 2D 생성 완료!', true);
          console.log('✅ 2D 생성 응답:', data);

          //viser 새 창 띄우기
          if (data.viser_result && data.viser_result.status) {
            const match = data.viser_result.status.match(/"(https:\/\/[^\s"]+)"/);
            if (match && match[1]) {
              window.open(match[1], '_blank');   // 👉 새 창으로 자동 열기
            }
          }

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

  // 상태 메시지 표시 함수
  function showStatus(message, isSuccess = true) {
    const statusEl = document.getElementById('status');
    statusEl.textContent = message;
    statusEl.className = isSuccess ? 'success' : 'error';
    statusEl.style.display = 'block';
    setTimeout(() => { statusEl.style.display = 'none'; }, 3000);
  }

  // 리뷰 요약 버튼 초기화 및 클릭 이벤트
  if (analyzeBtn) {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      // Airbnb 숙소 페이지에서만 버튼 활성화
      if (tabs[0].url.includes('airbnb.co.kr/rooms/')) analyzeBtn.disabled = false;
    });

    analyzeBtn.addEventListener('click', () => {
      clickSound.currentTime = 0;
      clickSound.play();

      analyzeBtn.disabled = true;
      // 버튼 텍스트 변경
      analyzeBtn.textContent = '분석 중...';

      chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        const url = tabs[0].url;
        // Airbnb 숙소 페이지 확인 및 오류 처리
        if (!url.includes('airbnb.co.kr/rooms/')) {
          alert('Airbnb 숙소 상세 페이지에서만 사용할 수 있습니다.');
          analyzeBtn.disabled = false;
          analyzeBtn.textContent = '댓글 요약 시작';
          return;
        }

        // 서버에 리뷰 분석 요청 전송
        fetch(`${SERVER_BASE_URL}/analyze_review`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ url })
        })
          .then(res => res.json())
          .then(data => {
            // 분석 결과 URL이 있으면 새 탭으로 열기, 아니면 오류 표시
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

  // 기능 토글 버튼 (확장 기능 활성/비활성)
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

  // 페이지 이동 버튼 (팝업 간 이동)
  if (gotoBtn) gotoBtn.addEventListener('click', () => window.location.href = 'popup2.html');
  if (backBtn) backBtn.addEventListener('click', () => window.location.href = 'popup.html');
});
