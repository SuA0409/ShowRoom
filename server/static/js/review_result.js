// ========================================
// 📦 토픽 필터 버튼 클릭 로직
// - 버튼 클릭 시: 모든 버튼의 active 클래스 제거 후
//   클릭된 버튼만 active 클래스 추가
// - 선택된 토픽과 일치하는 카드만 표시
// ========================================
document.querySelectorAll('.filter-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    // 모든 버튼에서 'active' 클래스 제거 후 클릭된 버튼만 활성화
    document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');

    const selectedTopic = btn.dataset.topic;

    // 선택된 토픽과 일치하는 카드만 보이도록 설정
    document.querySelectorAll('.topic-card').forEach(card => {
      card.style.display = (selectedTopic === 'All' || card.dataset.topic === selectedTopic)
        ? 'block'
        : 'none';
    });
  });
});


// ========================================
// 📦 “더 보기” 버튼 클릭 로직
// - 5개씩 추가로 항목을 표시
// - 더 이상 표시할 항목이 없으면 버튼 숨김 처리
// - 접기 버튼은 보이도록 처리
// ========================================
function expandList(index) {
  const listItems = document.querySelectorAll(`#list-${index} li`);
  const btnExpand = document.getElementById(`btn-expand-${index}`);
  const btnCollapse = document.getElementById(`btn-collapse-${index}`);

  // 현재 보이는 개수와 전체 개수 계산
  let shownCount = parseInt(btnExpand.dataset.shown);
  const totalCount = listItems.length;
  const nextCount = Math.min(shownCount + 5, totalCount);

  // 추가로 5개 더 보여줌
  listItems.forEach((item, idx) => {
    if (idx < nextCount) item.style.display = "list-item";
  });

  // 상태 업데이트
  btnExpand.dataset.shown = nextCount;
  btnCollapse.classList.remove("hidden");
  if (nextCount >= totalCount) {
    btnExpand.style.display = "none";
  }
}


// ========================================
// 📦 “접기” 버튼 클릭 로직
// - 다시 처음 5개만 보이도록 초기화
// - 더 보기 버튼 다시 활성화
// - 접기 버튼 숨김 처리
// ========================================
function collapseList(index) {
  const listItems = document.querySelectorAll(`#list-${index} li`);
  const btnExpand = document.getElementById(`btn-expand-${index}`);
  const btnCollapse = document.getElementById(`btn-collapse-${index}`);

  // 처음 5개만 보이게, 나머지는 숨김 처리
  listItems.forEach((item, idx) => {
    item.style.display = idx < 5 ? "list-item" : "none";
  });

  // 버튼 상태 초기화
  btnExpand.style.display = "block";
  btnExpand.dataset.shown = 5;
  btnCollapse.classList.add("hidden");
}


// ========================================
// 📦 페이지 로드 시 초기 상태 설정
// - 각 리뷰 리스트에서 6번째 이후 항목 숨김 처리
// ========================================
document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll("ul[id^='list-']").forEach(list => {
    const items = list.querySelectorAll("li");
    items.forEach((item, idx) => {
      if (idx >= 5) item.style.display = "none";
    });
  });
});
