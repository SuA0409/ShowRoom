document.addEventListener('DOMContentLoaded', () => {
  // 클릭 효과음 오디오 요소 가져오기
  const clickSound = document.getElementById('click-sound');

  // ── 0) 팝업 닫기 버튼 동작 ──
  const closeBtn = document.getElementById('close-popup');
  if (closeBtn) {
    closeBtn.addEventListener('click', () => {
      window.close();  // 팝업 창을 닫음
    });
  }

  // ── 0-2) ShowRoom 로고 버튼 클릭 시 이동 URL 설정 ──
  const logoBtn = document.getElementById('logo-button');
  if (logoBtn) {
    logoBtn.addEventListener('click', () => {
      clickSound.currentTime = 0;
      clickSound.play();  // 클릭 효과음 재생
      window.open('https://your-domain/showroom', '_blank');  // 새 탭으로 ShowRoom 페이지 열기
    });
  }

  // ── 1) 3D 변환 페이지 전용 요소 ──
  const convertBtn = document.getElementById('convert');  // 3D 변환 버튼
  const status = document.getElementById('status');      // 상태 메시지 영역

  // ── 2) 댓글 요약 페이지 전용 요소 ──
  const analyzeBtn = document.getElementById('analyze-review');  // 댓글 분석 버튼

  // ── 3) 공통 요소 ──
  const toggleBtn = document.getElementById('toggle');         // 기능 토글 버튼
  const gotoBtn = document.getElementById('goto-comments');    // 3D → 댓글 화면 이동 버튼
  const backBtn = document.getElementById('back');             // 댓글 → 3D 화면 이동 버튼

  // ── [추가] 썸네일 렌더링 함수 ──
  function renderThumbnails(images) {
    const container = document.getElementById('selected-thumbnails');
    const countEl = document.getElementById('selected-count');
    if (!container) return;

    // 1) 썸네일 영역 비우기
    container.innerHTML = '';

    // 2) 선택된 이미지 개수 표시
    if (countEl) {
      countEl.textContent = `선택된 이미지: ${images.length}장`;
    }

    // 3) 각 이미지 URL마다 썸네일 요소 생성
    images.forEach(src => {
      const img = document.createElement('img');
      img.src = src;  // 썸네일 이미지 소스 설정
      img.addEventListener('click', () => {
        // 썸네일 클릭 시 이미지 선택 해제 요청
        chrome.tabs.sendMessage(tabs[0].id, { action: 'toggleImage', src }, (newImages) => {
          renderThumbnails(newImages);  // 변경된 목록으로 다시 렌더링
        });
      });
      container.appendChild(img);  // 컨테이너에 썸네일 추가
    });
  }

  // ── 4) 3D 변환 페이지 로직 ──
  if (convertBtn) {
    // 변환 버튼 래퍼(부모 요소) 클릭해도 효과음 재생
    const wrapper = convertBtn.parentElement;
    wrapper.addEventListener('click', () => {
      clickSound.currentTime = 0;
      clickSound.play();
    });

    // 현재 탭이 Airbnb 페이지인지 확인 후 이미지 목록 요청
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      const tab = tabs[0];
      // Airbnb 도메인이 아니면 경고
      if (!tab.url.includes('airbnb.co.kr') && !tab.url.includes('airbnb.com')) {
        alert('Airbnb 상세 페이지에서 실행해주세요.');
        return;
      }

      // 콘텐츠 스크립트에 이미지 리스트 요청
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

        // 썸네일 및 선택 개수 표시
        renderThumbnails(response);
        convertBtn.disabled = response.length < 3;  // 3개 미만이면 버튼 비활성화
        setupConvertButton(tab.id);  // 버튼 클릭 핸들러 설정
      });
    });
  }

  // 3D 변환 버튼 클릭 이벤트 설정 함수
  function setupConvertButton(tabId) {
    // 기존 리스너 제거를 위해 버튼 복제 후 교체
    convertBtn.replaceWith(convertBtn.cloneNode(true));
    const newBtn = document.getElementById('convert');
    newBtn.disabled = false;

    newBtn.addEventListener('click', () => {
      clickSound.currentTime = 0;
      clickSound.play();  // 클릭 효과음 재생

      // 다시 선택된 이미지 목록 요청
      chrome.tabs.sendMessage(tabId, { action: 'getImages' }, async (images) => {
        if (chrome.runtime.lastError) {
          console.error('선택된 이미지 요청 실패:', chrome.runtime.lastError.message);
          alert('선택된 이미지를 다시 가져올 수 없습니다.');
          return;
        }
        if (!Array.isArray(images) || images.length < 3) {
          alert('최소 3장의 이미지를 선택해야 합니다.');
          return;
        }

        // 1) 서버 전송 전 썸네일 재렌더링
        renderThumbnails(images);

        // 2) 서버로 POST 요청 보내기
        try {
          const res = await fetch('https://86e4-163-152-3-173.ngrok-free.app/upload', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              'ngrok-skip-browser-warning': 'true'
            },
            body: JSON.stringify({ images })
          });
          const data = await res.json();

          if (data.view_url) {
            showStatus('✅ 3D 변환 요청이 성공적으로 전송되었습니다!', true);
            window.open(data.view_url, '_blank');  // 변환된 결과 URL 새 창 열기
          } else {
            showStatus('❌ 3D 변환 요청 중 오류가 발생했습니다.', false);
          }
        } catch (err) {
          console.error('3D 변환 요청 중 오류:', err);
          showStatus('❌ 서버 통신 오류가 발생했습니다.', false);
        }
      });
    });
  }

  // ── 5) 상태 메시지 표시 함수 ──
  function showStatus(message, isSuccess = true) {
    const statusEl = document.getElementById('status');
    statusEl.textContent = message;              // 메시지 설정
    statusEl.className = isSuccess ? 'success' : 'error'; // 클래스 지정
    statusEl.style.display = 'block';            // 표시
    setTimeout(() => {
      statusEl.style.display = 'none';           // 3초 후 숨김
    }, 3000);
  }

  // ── 6) 댓글 요약 페이지 로직 ──
  if (analyzeBtn) {
    // Airbnb 숙소 상세 페이지인지 확인 후 버튼 활성화
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      const tab = tabs[0];
      if (tab.url.includes('airbnb.co.kr/rooms/')) {
        analyzeBtn.disabled = false;
      }
    });

    // 댓글 분석 버튼 클릭 시 동작
    analyzeBtn.addEventListener('click', () => {
      clickSound.currentTime = 0;
      clickSound.play();  // 클릭 효과음 재생

      analyzeBtn.disabled = true;           // 버튼 비활성화
      analyzeBtn.textContent = '분석 중...'; // 버튼 텍스트 변경

      chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        const url = tabs[0].url;
        // Airbnb 숙소 상세 페이지가 아니면 경고하고 초기화
        if (!url.includes('airbnb.co.kr/rooms/')) {
          alert('Airbnb 숙소 상세 페이지에서만 사용할 수 있습니다.');
          analyzeBtn.disabled = false;
          analyzeBtn.textContent = '댓글 요약 시작';
          return;
        }

        // 서버로 댓글 분석 요청
        fetch('https://86e4-163-152-3-173.ngrok-free.app/analyze_review', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ url })
        })
        .then(res => res.json())
        .then(data => {
          if (data.view_url) {
            window.open(data.view_url, '_blank');  // 분석 결과 URL 열기
          } else {
            showStatus('❌ 분석 실패: ' + (data.error || 'Unknown error'), false);
          }
        })
        .catch(e => {
          showStatus('❌ 서버 오류: ' + e, false);
        })
        .finally(() => {
          analyzeBtn.disabled = false;            // 버튼 재활성화
          analyzeBtn.textContent = '댓글 요약 시작'; // 버튼 텍스트 복원
        });
      });
    });
  }

  // ── 7) 기능 토글 버튼 로직 ──
  if (toggleBtn) {
    // 저장된 상태 가져와 버튼 텍스트 설정
    chrome.storage.local.get('enabled', data => {
      const isOn = !!data.enabled;
      toggleBtn.textContent = isOn ? '기능 끄기' : '기능 켜기';
    });

    // 토글 버튼 클릭 시 상태 저장 및 콘텐츠 스크립트에 전파
    toggleBtn.addEventListener('click', () => {
      chrome.storage.local.get('enabled', data => {
        const newState = !data.enabled;
        chrome.storage.local.set({ enabled: newState }, () => {
          toggleBtn.textContent = newState ? '기능 끄기' : '기능 켜기';
          chrome.tabs.query({ active: true, currentWindow: true }, tabs => {
            // 콘텐츠 스크립트에 활성화 여부 메시지 전송
            chrome.tabs.sendMessage(tabs[0].id, { action: 'toggle', enabled: newState });
          });
        });
      });
    });
  }

  // ── 8) 페이지 간 네비게이션 (댓글 ↔ 3D) ──
  if (gotoBtn) {
    gotoBtn.addEventListener('click', () => {
      window.location.href = 'popup2.html';  // 3D → 댓글 화면 이동
    });
  }
  if (backBtn) {
    backBtn.addEventListener('click', () => {
      window.location.href = 'popup.html';   // 댓글 → 3D 화면 이동
    });
  }
});