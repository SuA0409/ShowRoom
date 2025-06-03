// ── 1) 확장 아이콘 클릭 시 별도 팝업 창(popup type) 열기 ──
chrome.action.onClicked.addListener((tab) => {
  chrome.windows.create({
    url: chrome.runtime.getURL('popup.html'), // 확장 프로그램 내부의 popup.html 파일 경로
    type: 'popup',                            // 팝업 형태로 창 생성
    width: 400,                               // 팝업 창 너비 설정
    height: 500,                              // 팝업 창 높이 설정
    top: 100,                                 // 화면 위쪽으로부터의 위치
    left: 100                                 // 화면 왼쪽으로부터의 위치
  });
});

// ── 2) 설치 시 컨텍스트 메뉴(“기능 켜기/끄기”) 생성 ──
chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "toggle-feature",                         // 메뉴 식별용 ID
    title: "Airbnb 이미지 기능 켜기/끄기",         // 우클릭 메뉴에 표시될 텍스트
    contexts: ["action"]                          // 확장 아이콘 우클릭시만 표시
  });
  chrome.storage.local.set({ enabled: false });   // 기본 상태: 기능 비활성화 저장
});

// ── 3) 컨텍스트 메뉴 클릭 시 “이미지 기능” 토글 메시지 전송 ──
chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === "toggle-feature") {
    // 저장된 상태를 가져와서 반전시킨 후 다시 저장
    chrome.storage.local.get('enabled', (data) => {
      const newState = !data.enabled;
      chrome.storage.local.set({ enabled: newState }, () => {
        // 현재 활성 탭에 메시지 전송해 기능 켜기/끄기 상태 전달
        chrome.tabs.sendMessage(tab.id, {
          action: 'toggle',
          enabled: newState
        });
      });
    });
  }
});
