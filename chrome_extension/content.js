(() => {
  // 중복 실행 방지
  if (window.hasRunContentScript) return;
  window.hasRunContentScript = true;

  // 선택된 이미지 URL 배열, enabled = 기능 활성화 여부
  let selectedImages = [];
  let enabled = false;

  // 초기 상태 로딩 (토글 상태 및 이미지 목록)
  chrome.storage.local.get(['enabled', 'selectedImages'], (data) => {
    enabled = data.enabled ?? false;
    selectedImages = data.selectedImages ?? [];
    highlightSelectedImages();
  });

  // 메시지 수신 처리 (toggle/getImages/sendToServer)
  chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
    if (msg.action === 'toggle') {
      enabled = msg.enabled;
      if (!enabled) {
        clearHighlights();
        selectedImages = [];
        chrome.storage.local.set({ selectedImages: [] });
      }
      sendResponse({ success: true });
      return true;
    }
    
    // 선택된 이미지 배열 반환
    if (msg.action === 'getImages') {
      sendResponse(selectedImages);
      return true;
    }

    // 이미지 배열 서버로 전송
    if (msg.action === 'sendToServer') {
      sendImagesToServer(selectedImages);
      sendResponse({ sent: true });
      return true;
    }
  });

  // 이미지 클릭 이벤트 (capture 단계)
  document.addEventListener('click', (e) => {
    if (!enabled) return;
    const container = e.target.closest('img, picture, div[role="button"], div[role="dialog"]');
    if (!container) return;
    const src = extractImageSrc(container);
    if (!src) return;

    e.preventDefault();
    e.stopImmediatePropagation();
    toggleSelection(src);
  }, true);

  // 이미지 선택/해제 및 하이라이트 처리
  function toggleSelection(src) {
    const index = selectedImages.indexOf(src);
    if (index === -1) {
      selectedImages.push(src);
    } else {
      selectedImages.splice(index, 1);
    }
    chrome.storage.local.set({ selectedImages });
    highlightSelectedImages();
  }

  // 선택된 이미지는 빨간 테두리 표시, 나머지는 테두리 제거
  function highlightSelectedImages() {
    document.querySelectorAll('img').forEach((img) => {
      if (selectedImages.includes(img.src)) {
        img.style.setProperty('outline', '4px solid red', 'important');
        img.style.setProperty('outline-offset', '-4px', 'important');
      } else {
        img.style.setProperty('outline', '', 'important');
        img.style.setProperty('outline-offset', '', 'important');
      }
    });
  }

  // 모든 이미지의 테두리(하이라이트) 제거
  function clearHighlights() {
    document.querySelectorAll('img').forEach((img) => {
      img.style.setProperty('outline', '', 'important');
      img.style.setProperty('outline-offset', '', 'important');
    });
  }

  // 클릭된 요소에서 이미지 URL 추출
  function extractImageSrc(el) {
    if (el.tagName === 'IMG') return el.src;
    const img = el.querySelector('img');
    if (img && img.src) return img.src;
    const bg = el.style.backgroundImage;
    if (bg && bg.includes('url')) {
      const match = bg.match(/url\("?(.+?)"?\)/);
      return match ? match[1] : '';
    }
    const dataSrc = el.getAttribute('data-src') || el.getAttribute('src');
    return dataSrc || '';
  }

  // 선택 이미지 서버로 전송
  async function sendImagesToServer(images) {
    try {
      const response = await fetch('https://xxxx.ngrok-free.app/upload', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'ngrok-skip-browser-warning': 'true'
        },
        body: JSON.stringify({ images })
      });
      const result = await response.json();

      if (result.status === 'success' || result.view_url) {
        showSuccessMessage();
        if (result.view_url) window.open(result.view_url, '_blank');
      } else {
        alert('전송 실패: ' + (result.error || '알 수 없는 오류'));
      }
    } catch (err) {
      console.error('전송 중 오류:', err);
      alert('서버 통신 오류');
    }
  }

  // 전송 성공 메시지 표시
  function showSuccessMessage() {
    const msg = document.createElement('div');
    msg.innerText = '✔ 이미지 전송 완료!';
    Object.assign(msg.style, {
      position: 'fixed',
      top: '20px',
      right: '20px',
      background: '#28a745',
      color: 'white',
      padding: '10px 20px',
      borderRadius: '8px',
      zIndex: 9999,
      fontSize: '16px'
    });
    document.body.appendChild(msg);
    setTimeout(() => msg.remove(), 3000);
  }
})();
