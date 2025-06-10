// 확장 아이콘 클릭 시 팝업창 오픈
chrome.action.onClicked.addListener((tab) => {
  chrome.windows.create({
    url: chrome.runtime.getURL('popup.html'),
    type: 'popup',
    width: 400,
    height: 500,
    top: 100,
    left: 100
  });
});

// 확장 설치 시 컨텍스트 메뉴 추가 및 기능 상태 초기화
chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "toggle-feature",
    title: "Airbnb 이미지 기능 on/off",
    contexts: ["action"]
  });
  chrome.storage.local.set({ enabled: false });
});

// 컨텍스트 메뉴 클릭 시 기능 on/off 토글
chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === "toggle-feature") {
    chrome.storage.local.get('enabled', (data) => {
      const newState = !data.enabled;
      chrome.storage.local.set({ enabled: newState }, () => {
        chrome.tabs.sendMessage(tab.id, {
          action: 'toggle',
          enabled: newState
        });
      });
    });
  }
});