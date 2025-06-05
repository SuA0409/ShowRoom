// ===== 토픽 필터 로직 =====
document.querySelectorAll('.filter-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    // 버튼 활성 상태 토글
    document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');

    const selectedTopic = btn.dataset.topic;
    document.querySelectorAll('.topic-card').forEach(card => {
      card.style.display = (selectedTopic === 'All' || card.dataset.topic === selectedTopic)
        ? 'block'
        : 'none';
    });

    // 아무 스크롤 동작 없음 → 카드가 절대 위치 고정
  });
});

// ===== “더 보기” 기능 =====
function expandList(index) {
  const listItems = document.querySelectorAll(`#list-${index} li`);
  const btnExpand = document.getElementById(`btn-expand-${index}`);
  const btnCollapse = document.getElementById(`btn-collapse-${index}`);
  let shownCount = parseInt(btnExpand.dataset.shown);
  const totalCount = listItems.length;
  const nextCount = Math.min(shownCount + 5, totalCount);
  listItems.forEach((item, idx) => {
    if (idx < nextCount) item.style.display = "list-item";
  });
  btnExpand.dataset.shown = nextCount;
  btnCollapse.classList.remove("hidden");
  if (nextCount >= totalCount) {
    btnExpand.style.display = "none";
  }
}

// ===== “접기” 기능 =====
function collapseList(index) {
  const listItems = document.querySelectorAll(`#list-${index} li`);
  const btnExpand = document.getElementById(`btn-expand-${index}`);
  const btnCollapse = document.getElementById(`btn-collapse-${index}`);
  listItems.forEach((item, idx) => {
    item.style.display = idx < 5 ? "list-item" : "none";
  });
  btnExpand.style.display = "block";
  btnExpand.dataset.shown = 5;
  btnCollapse.classList.add("hidden");
}

// ===== 초기화 =====
document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll("ul[id^='list-']").forEach(list => {
    const items = list.querySelectorAll("li");
    items.forEach((item, idx) => {
      if (idx >= 5) item.style.display = "none";
    });
  });
});