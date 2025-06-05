(() => {
  // 이미 한 번 실행된 경우 반복 실행 방지
  if (window.hasRunContentScript) return;
  window.hasRunContentScript = true;

  // 선택된 이미지 URL 목록을 저장할 배열
  let selectedImages = [];
  // 기능 활성화 여부 (토글 상태)
  let enabled = false;

  // ── 초기 상태 로딩 ──
  chrome.storage.local.get(['enabled', 'selectedImages'], (data) => {
    enabled = data.enabled ?? false;                // 저장된 활성화 상태 불러오기 (기본 false)
    selectedImages = data.selectedImages ?? [];     // 저장된 선택 이미지 목록 불러오기 (기본 빈 배열)
    highlightSelectedImages();                      // 페이지 로드 후 이미 선택된 이미지에 테두리 표시
  });

  // ── 메시지 핸들링 (background 또는 popup에서 보내는 메시지 처리) ──
  chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
    // 기능 토글 요청 받으면 enabled 상태 변경 및 필요 시 테두리 제거
    if (msg.action === 'toggle') {
      enabled = msg.enabled;
      if (!enabled) {
        clearHighlights();                           // 비활성화 시 모든 테두리 제거
        selectedImages = [];                          // 선택 목록 초기화
        chrome.storage.local.set({ selectedImages: [] });
      }
      sendResponse({ success: true });
      return true;
    }

    // 선택된 이미지 목록을 요청받으면 응답으로 보내줌
    if (msg.action === 'getImages') {
      sendResponse(selectedImages);
      return true;
    }

    // 서버 전송 요청이 오면 sendImagesToServer 호출
    if (msg.action === 'sendToServer') {
      sendImagesToServer(selectedImages);
      sendResponse({ sent: true });
      return true;
    }
  });

  // ── 이미지 클릭 이벤트 (capture 단계에서 먼저 가로채기) ──
  document.addEventListener('click', (e) => {
    if (!enabled) return;  // 기능이 꺼져 있으면 아무 동작도 하지 않음

    // 클릭된 요소가 이미지 혹은 이미지가 포함된 컨테이너인지 찾기
    const container = e.target.closest('img, picture, div[role="button"], div[role="dialog"]');
    if (!container) return;

    // 실제 이미지 URL을 추출
    const src = extractImageSrc(container);
    if (!src) return;

    // 기본 클릭 동작 방지 (Airbnb의 링크 이동 등 차단)
    e.preventDefault();
    e.stopImmediatePropagation();

    // 선택 토글 기능 호출
    toggleSelection(src);
  }, true); // true를 주어 capture 단계에서 먼저 실행

  // ── 선택 토글 함수: 이미지가 목록에 없으면 추가, 있으면 제거 ──
  function toggleSelection(src) {
    const index = selectedImages.indexOf(src);
    if (index === -1) {
      selectedImages.push(src);
    } else {
      selectedImages.splice(index, 1);
    }
    // 변경된 목록을 저장소에 업데이트
    chrome.storage.local.set({ selectedImages });
    // 테두리 새로 표시
    highlightSelectedImages();
  }

  // ── 선택된 이미지에 빨간 테두리 표시 ──
  function highlightSelectedImages() {
    document.querySelectorAll('img').forEach((img) => {
      const src = img.src;
      if (selectedImages.includes(src)) {
        // 선택된 이미지인 경우 강제적으로 테두리 스타일 적용
        img.style.setProperty('outline', '4px solid red', 'important');
        img.style.setProperty('outline-offset', '-4px', 'important');
      } else {
        // 선택되지 않은 경우 테두리 제거
        img.style.setProperty('outline', '', 'important');
        img.style.setProperty('outline-offset', '', 'important');
      }
    });
  }

  // ── 모든 이미지의 테두리를 제거 ──
  function clearHighlights() {
    document.querySelectorAll('img').forEach((img) => {
      img.style.setProperty('outline', '', 'important');
      img.style.setProperty('outline-offset', '', 'important');
    });
  }

  // ── 클릭된 요소에서 이미지 URL 추출 (IMG 태그, background-image, data-src 등) ──
  function extractImageSrc(el) {
    // 1) <img> 태그일 경우 src 직접 반환
    if (el.tagName === 'IMG') return el.src;

    // 2) 자식 요소에 <img>가 있을 때 해당 src 반환
    const img = el.querySelector('img');
    if (img && img.src) return img.src;

    // 3) background-image 스타일에 URL이 있을 경우 추출
    const bg = el.style.backgroundImage;
    if (bg && bg.includes('url')) {
      const match = bg.match(/url\("?(.+?)"?\)/);
      return match ? match[1] : '';
    }

    // 4) data-src나 src 속성으로 이미지 경로 저장한 경우
    const dataSrc = el.getAttribute('data-src') || el.getAttribute('src');
    return dataSrc || '';
  }

  // ── 선택된 이미지 목록을 서버로 전송 ──
  async function sendImagesToServer(images) {
    try {
      const response = await fetch('https://xxxx.ngrok-free.app/upload', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'ngrok-skip-browser-warning': 'true'
        },
        body: JSON.stringify({ images })  // 이미지 URL 배열을 JSON으로 전송
      });

      const result = await response.json();

      if (result.status === 'success' || result.view_url) {
        showSuccessMessage();
        // 서버에서 3D 뷰 URL을 반환하면 새 탭으로 열기
        if (result.view_url) window.open(result.view_url, '_blank');
      } else {
        alert('전송 실패: ' + (result.error || '알 수 없는 오류'));
      }
    } catch (err) {
      console.error('전송 중 오류:', err);
      alert('서버 통신 오류');
    }
  }

  // ── 이미지 전송 성공 시 페이지 상단에 알림 메시지 표시 ──
  function showSuccessMessage() {
    const msg = document.createElement('div');
    msg.innerText = '✔ 이미지 전송 완료!';
    Object.assign(msg.style, {
      position: 'fixed',
      top: '20px',
      right: '20px',
      background: '#28a745',   // 녹색 배경
      color: 'white',
      padding: '10px 20px',
      borderRadius: '8px',
      zIndex: 9999,
      fontSize: '16px'
    });
    document.body.appendChild(msg);
    setTimeout(() => msg.remove(), 3000);  // 3초 후 메시지 제거
  }
})();