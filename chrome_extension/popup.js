document.addEventListener('DOMContentLoaded', () => {
  const clickSound = document.getElementById('click-sound');

  // ── [공통 서버 주소 상수] ──
  const SERVER_BASE_URL = 'https://0aaa-34-87-21-83.ngrok-free.app';

  // ── 0) 닫기 버튼 동작 ──
  const closeBtn = document.getElementById('close-popup');
  if (closeBtn) {
    closeBtn.addEventListener('click', () => {
      window.close();
    });
  }

  // ── 0-2) ShowRoom 로고 버튼 클릭 시 이동할 URL 설정 ──
  const logoBtn = document.getElementById('logo-button');
  if (logoBtn) {
    logoBtn.addEventListener('click', () => {
      clickSound.currentTime = 0;
      clickSound.play();
      window.open('https://your-domain/showroom', '_blank');
    });
  }

  // ── 1) 3D 변환 페이지 전용 요소 ──
  const convertBtn = document.getElementById('convert');
  const status = document.getElementById('status');

  // ── 2) 댓글 요약 페이지 전용 요소 ──
  const analyzeBtn = document.getElementById('analyze-review');

  // ── 3) 공통 요소 ──
  const toggleBtn = document.getElementById('toggle');
  const gotoBtn = document.getElementById('goto-comments');
  const backBtn = document.getElementById('back');

  // ── 썸네일 렌더링 함수 ──
  function renderThumbnails(images) {
    const container = document.getElementById('selected-thumbnails');
    const countEl = document.getElementById('selected-count');
    if (!container) return;

    container.innerHTML = '';
    if (countEl) {
      countEl.textContent = `선택된 이미지: ${images.length}장`;
    }

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

  // ── 4) 3D 변환 페이지 로직 ──
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

  function setupConvertButton(tabId) {
    convertBtn.replaceWith(convertBtn.cloneNode(true));
    const newBtn = document.getElementById('convert');
    newBtn.disabled = false;

    newBtn.addEventListener('click', () => {
      clickSound.currentTime = 0;
      clickSound.play();

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

        renderThumbnails(images);

        try {
          const res = await fetch(`${SERVER_BASE_URL}/upload`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              'ngrok-skip-browser-warning': 'true'
            },
            body: JSON.stringify({ images })
          });
          const data = await res.json();

          if (data.status === 'success') {
            showStatus('✅ 3D 변환 요청이 성공적으로 전송되었습니다!', true);
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
    statusEl.textContent = message;
    statusEl.className = isSuccess ? 'success' : 'error';
    statusEl.style.display = 'block';
    setTimeout(() => {
      statusEl.style.display = 'none';
    }, 3000);
  }

  // ── 6) 댓글 요약 페이지 로직 ──
  if (analyzeBtn) {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      const tab = tabs[0];
      if (tab.url.includes('airbnb.co.kr/rooms/')) {
        analyzeBtn.disabled = false;
      }
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
          if (data.view_url) {
            window.open(data.view_url, '_blank');
          } else {
            showStatus('❌ 분석 실패: ' + (data.error || 'Unknown error'), false);
          }
        })
        .catch(e => {
          showStatus('❌ 서버 오류: ' + e, false);
        })
        .finally(() => {
          analyzeBtn.disabled = false;
          analyzeBtn.textContent = '댓글 요약 시작';
        });
      });
    });
  }

  // ── 7) 기능 토글 버튼 로직 ──
  if (toggleBtn) {
    chrome.storage.local.get('enabled', data => {
      const isOn = !!data.enabled;
      toggleBtn.textContent = isOn ? 'off' : 'on';
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

  // ── 8) 페이지 간 네비게이션(댓글→3D, 3D→댓글) ──
  if (gotoBtn) {
    gotoBtn.addEventListener('click', () => {
      window.location.href = 'popup2.html';
    });
  }
  if (backBtn) {
    backBtn.addEventListener('click', () => {
      window.location.href = 'popup.html';
    });
  }
});
