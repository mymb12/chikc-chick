// --- START OF FILE script.js ---
document.addEventListener("DOMContentLoaded", () => {
  // --- Element Selections ---
  const navButtons = document.querySelectorAll(".bottom-nav .nav-button");
  const tabPanes = document.querySelectorAll(".tab-pane");
  const normalFeedImg = document.getElementById("normal-feed-img");
  const diseaseList = document.getElementById("disease-list");
  const diseaseDetailsPane = document.getElementById("disease-details");
  const diseaseDetailTitle = document.getElementById("disease-detail-title");
  const diseaseSymptoms = document.getElementById("disease-symptoms");
  const diseaseCure = document.getElementById("disease-cure");
  const diseasePrevention = document.getElementById("disease-prevention");
  const reportsBell = document.getElementById("reports-bell");
  const reportsListPane = document.getElementById("reports-list");
  const reportsContent = document.getElementById("reports-content");
  const toggleFeedBtn = document.getElementById("toggle-feed-btn");

  // --- Constants ---
  const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${wsProtocol}//${window.location.host}/ws`;

  // --- State ---
  let socket = null;
  let currentVideoUrl = null;
  let activeTabId =
    document.querySelector(".tab-pane.active")?.id || "home-pane";
  let reconnectAttempts = 0;
  const maxReconnectAttempts = 5;
  const reconnectDelayBase = 3000;
  let reconnectTimer = null;
  let isLiveFeedActive = true;

  // --- Simulated Data ---
  const diseaseData = {
    /* ... (keep existing disease data) ... */
    coccidiosis: {
      name: "Кокцидиоз",
      symptoms:
        "Кровавый понос, вялость, взъерошенность оперения, снижение аппетита и веса.",
      cure: "Применение кокцидиостатиков (например, ампролиум, сульфаниламиды) согласно инструкции. Поддерживающая терапия.",
      prevention:
        "Соблюдение гигиены в птичнике, регулярная чистка и дезинфекция, использование качественных кормов, недопущение скученности.",
    },
    mareks: {
      name: "Болезнь Марека",
      symptoms:
        "Параличи конечностей (одной или обеих), крыльев, шеи. Опухоли внутренних органов, кожи, мышц. Помутнение глаз (сероглазка).",
      cure: "Эффективного лечения нет. Больных птиц выбраковывают.",
      prevention:
        "Вакцинация цыплят в суточном возрасте. Строгие ветеринарно-санитарные меры.",
    },
    newcastle: {
      name: "Болезнь Ньюкасла",
      symptoms:
        "Респираторные признаки (кашель, чихание, хрипы), диарея (зеленоватый помет), нервные явления (скручивание шеи, параличи), снижение яйценоскости.",
      cure: "Специфического лечения нет. Больных птиц уничтожают.",
      prevention:
        "Вакцинация поголовья. Карантин для новой птицы. Санитарный контроль.",
    },
    pullorosis: {
      name: "Пуллороз (Белый бациллярный понос)",
      symptoms:
        "У цыплят: слабость, белый или желтоватый пенистый понос, склеивание пуха вокруг клоаки, высокая смертность. У взрослых: снижение яйценоскости, воспаление яичников.",
      cure: "Антибиотики (фуразолидон, левомицетин) и сульфаниламиды. Лечение малоэффективно для цыплят.",
      prevention:
        "Приобретение инкубационных яиц и цыплят из благополучных хозяйств. Анализ крови у взрослого поголовья.",
    },
    pasteurellosis: {
      name: "Пастереллёз (Холера птиц)",
      symptoms:
        "Острая форма: внезапная гибель без видимых признаков. Подострая/хроническая: вялость, синюшность гребня и сережек, диарея, артриты, ринит, конъюнктивит.",
      cure: "Антибиотики (тетрациклины, сульфаниламиды). Лечение эффективно на ранних стадиях.",
      prevention: "Вакцинация. Борьба с грызунами. Соблюдение санитарных норм.",
    },
  };

  function generateSimulatedReports() {
    /* ... (keep existing report generation) ... */
    const reports = [];
    const statuses = [
      { text: "Здоровье", class: "status-healthy" },
      { text: "Предупреждение", class: "status-warning" },
      { text: "Опасность", class: "status-danger" },
    ];
    const issues = [
      "Отклонений нет",
      "Небольшая вариация температуры",
      "Подозрительная активность",
      "Обнаружена высокая температура",
      "Замечено аномальное поведение",
      "Снижение подвижности в зоне",
      "Скученность птиц",
    ];
    const reportCount = Math.floor(Math.random() * 5) + 3;
    for (let i = 0; i < reportCount; i++) {
      const randomStatusIndex = Math.floor(Math.random() * statuses.length);
      const randomStatus = statuses[randomStatusIndex];
      let randomIssue;
      if (randomStatus.class === "status-healthy") {
        randomIssue = issues[0];
      } else if (randomStatus.class === "status-warning") {
        const warningIssues = issues.filter(
          (issue) =>
            issue.includes("Небольшая") ||
            issue.includes("Подозрительная") ||
            issue.includes("Снижение") ||
            issue.includes("Скученность"),
        );
        randomIssue =
          warningIssues[Math.floor(Math.random() * warningIssues.length)];
      } else {
        const dangerIssues = issues.filter(
          (issue) =>
            issue.includes("Обнаружена") || issue.includes("аномальное"),
        );
        randomIssue =
          dangerIssues[Math.floor(Math.random() * dangerIssues.length)];
      }
      const date = new Date(
        Date.now() - i * Math.random() * 1000 * 60 * 60 * 18,
      );
      reports.push({
        id: `rep_${Date.now()}_${i}`,
        timestamp: date.toLocaleString("ru-RU", {
          day: "2-digit",
          month: "2-digit",
          year: "numeric",
          hour: "2-digit",
          minute: "2-digit",
        }),
        status: randomStatus.text,
        statusClass: randomStatus.class,
        details: `${randomIssue}. Зона: ${Math.floor(Math.random() * 3) + 1}.`,
      });
    }
    reports.sort(
      (a, b) =>
        new Date(
          b.timestamp.split(", ")[0].split(".").reverse().join("-") +
            "T" +
            b.timestamp.split(", ")[1],
        ) -
        new Date(
          a.timestamp.split(", ")[0].split(".").reverse().join("-") +
            "T" +
            a.timestamp.split(", ")[1],
        ),
    );
    return reports;
  }

  // --- WebSocket Functions ---
  function connectWebSocket() {
    if (
      socket &&
      (socket.readyState === WebSocket.OPEN ||
        socket.readyState === WebSocket.CONNECTING)
    )
      return;
    if (reconnectTimer) clearTimeout(reconnectTimer);
    console.log(
      `%c[WS] Attempting connection to: ${wsUrl} (Attempt ${reconnectAttempts + 1})`,
      "color: blue",
    );
    setVideoFeedStatus("Подключение WebSocket...");
    if (currentVideoUrl) {
      URL.revokeObjectURL(currentVideoUrl);
      currentVideoUrl = null;
    }
    socket = new WebSocket(wsUrl);

    socket.onopen = (event) => {
      console.log("%c[WS] Connection established.", "color: green");
      reconnectAttempts = 0;
      if (activeTabId === "camera-pane") {
        const streamToSubscribe = isLiveFeedActive
          ? "normal_video"
          : "static_video";
        // <<< MODIFIED Alt Text >>>
        const statusMsg = isLiveFeedActive
          ? "Запрос live потока (детектор движения)..."
          : "Запрос video потока (детектор движения)...";
        setVideoFeedStatus(statusMsg);
        sendWebSocketMessage({
          action: "subscribe",
          stream: streamToSubscribe,
        });
      } else {
        setVideoFeedStatus("");
      }
    };

    socket.onmessage = (event) => {
      if (event.data instanceof Blob) {
        if (activeTabId === "camera-pane") {
          const newUrl = URL.createObjectURL(event.data);
          if (normalFeedImg) {
            if (currentVideoUrl) {
              URL.revokeObjectURL(currentVideoUrl);
            }
            normalFeedImg.src = newUrl;
            currentVideoUrl = newUrl;
            // <<< MODIFIED Alt Text >>>
            normalFeedImg.alt = isLiveFeedActive
              ? "Прямая трансляция (Детектор Движения)"
              : "Обработанное видео (Детектор Движения)";
          } else {
            URL.revokeObjectURL(newUrl);
          }
        } else {
          const tempUrl = URL.createObjectURL(event.data);
          URL.revokeObjectURL(tempUrl);
        }
      } else if (typeof event.data === "string") {
        console.log("[WS] Received text message:", event.data);
      } else {
        console.warn(
          "[WS] Received unexpected message type:",
          typeof event.data,
        );
      }
    };

    socket.onerror = (event) => {
      console.error("[WS] WebSocket error:", event);
      if (activeTabId === "camera-pane")
        setVideoFeedStatus("Ошибка WebSocket.");
    };

    socket.onclose = (event) => {
      console.log(
        `%c[WS] Connection closed: Code=${event.code}, Reason='${event.reason || "N/A"}' Clean=${event.wasClean}`,
        "color: orange",
      );
      socket = null;
      if (currentVideoUrl) {
        URL.revokeObjectURL(currentVideoUrl);
        currentVideoUrl = null;
        console.log("[WS] Revoked Blob URL on close.");
      }
      if (activeTabId === "camera-pane") {
        setVideoFeedStatus(`WebSocket отключен (Код: ${event.code})`);
        if (normalFeedImg) {
          normalFeedImg.src = "";
          normalFeedImg.alt = `WebSocket отключен`;
        }
      }
      if (reconnectAttempts < maxReconnectAttempts) {
        reconnectAttempts++;
        const delay = reconnectDelayBase * Math.pow(1.5, reconnectAttempts - 1);
        console.log(
          `[WS] Attempting reconnect #${reconnectAttempts} in ${delay / 1000}s...`,
        );
        if (activeTabId === "camera-pane")
          setVideoFeedStatus(`Переподключение #${reconnectAttempts}...`);
        if (reconnectTimer) clearTimeout(reconnectTimer);
        reconnectTimer = setTimeout(connectWebSocket, delay);
      } else {
        console.error("[WS] Max reconnection attempts reached.");
        if (activeTabId === "camera-pane")
          setVideoFeedStatus("Не удалось подключиться.");
      }
    };
  }

  function sendWebSocketMessage(message) {
    if (socket && socket.readyState === WebSocket.OPEN) {
      try {
        socket.send(JSON.stringify(message));
        console.log("[WS] Sent:", message);
      } catch (e) {
        console.error("[WS] Error sending message:", e);
      }
    } else {
      console.warn(
        `[WS] Cannot send, WebSocket not open. State: ${socket?.readyState}. Msg:`,
        message,
      );
    }
  }

  function setVideoFeedStatus(statusText) {
    if (normalFeedImg && activeTabId === "camera-pane") {
      if (!currentVideoUrl && !normalFeedImg.src) {
        normalFeedImg.alt = statusText;
      } else if (!normalFeedImg.src) {
        // <<< MODIFIED Alt Text >>>
        normalFeedImg.alt = isLiveFeedActive
          ? "Прямая трансляция (Детектор Движения)"
          : "Обработанное видео (Детектор Движения)";
      }
    } else if (normalFeedImg && activeTabId !== "camera-pane") {
      if (!normalFeedImg.src) {
        normalFeedImg.alt = "Камера не активна";
      }
    }
  }

  // --- Tab Switching Logic ---
  navButtons.forEach((button) => {
    button.addEventListener("click", () => {
      const targetPaneId = button.getAttribute("data-tab");
      if (button.classList.contains("active") || !targetPaneId) return;

      const previousActiveTabId = activeTabId;
      navButtons.forEach((btn) => btn.classList.remove("active"));
      tabPanes.forEach((pane) => pane.classList.remove("active"));
      button.classList.add("active");
      const targetPane = document.getElementById(targetPaneId);
      if (targetPane) targetPane.classList.add("active");
      activeTabId = targetPaneId;

      const currentStream = isLiveFeedActive ? "normal_video" : "static_video";
      if (
        targetPaneId === "camera-pane" &&
        previousActiveTabId !== "camera-pane"
      ) {
        console.log(
          `[Tab] Switched TO Camera (Mode: ${isLiveFeedActive ? "Live" : "Static"})`,
        );
        // <<< MODIFIED Alt Text >>>
        const statusMsg = isLiveFeedActive
          ? "Запрос live потока (детектор движения)..."
          : "Запрос video потока (детектор движения)...";
        setVideoFeedStatus(statusMsg);
        sendWebSocketMessage({ action: "subscribe", stream: currentStream });
        if (!socket || socket.readyState !== WebSocket.OPEN) {
          connectWebSocket();
        }
      } else if (
        targetPaneId !== "camera-pane" &&
        previousActiveTabId === "camera-pane"
      ) {
        console.log(
          `[Tab] Switched AWAY from Camera (Was: ${isLiveFeedActive ? "Live" : "Static"})`,
        );
        sendWebSocketMessage({ action: "unsubscribe", stream: currentStream });
        if (normalFeedImg) {
          normalFeedImg.src = "";
          normalFeedImg.alt = "Трансляция остановлена.";
          if (currentVideoUrl) {
            URL.revokeObjectURL(currentVideoUrl);
            currentVideoUrl = null;
          }
        }
        setVideoFeedStatus("");
      }
      hideOverlay(diseaseDetailsPane);
      hideOverlay(reportsListPane);
    });
  });

  // --- Info Pane & Overlay Logic ---
  function showOverlay(overlayElement) {
    /* ... (keep as is) ... */
    if (!overlayElement) {
      console.error("showOverlay: Element not found!");
      return;
    }
    overlayElement.style.display = "flex";
  }
  function hideOverlay(overlayElement) {
    /* ... (keep as is) ... */
    if (!overlayElement) return;
    overlayElement.style.display = "none";
  }
  if (diseaseList) {
    /* ... (keep as is) ... */
    diseaseList.addEventListener("click", (event) => {
      const listItem = event.target.closest(".disease-item");
      if (!listItem) return;
      const diseaseId = listItem.getAttribute("data-disease-id");
      const data = diseaseData[diseaseId];
      if (
        data &&
        diseaseDetailTitle &&
        diseaseSymptoms &&
        diseaseCure &&
        diseasePrevention &&
        diseaseDetailsPane
      ) {
        diseaseDetailTitle.textContent = data.name;
        diseaseSymptoms.textContent = data.symptoms;
        diseaseCure.textContent = data.cure;
        diseasePrevention.textContent = data.prevention;
        showOverlay(diseaseDetailsPane);
      } else {
        console.error("Missing elements/data for disease details");
      }
    });
  } else {
    console.error("Element with ID 'disease-list' not found!");
  }
  if (reportsBell) {
    /* ... (keep as is) ... */
    reportsBell.addEventListener("click", () => {
      if (!reportsContent || !reportsListPane) {
        console.error("Missing reports elements");
        return;
      }
      const simulatedReports = generateSimulatedReports();
      reportsContent.innerHTML = ""; // Clear previous
      if (simulatedReports.length > 0) {
        simulatedReports.forEach((report) => {
          const reportDiv = document.createElement("div");
          reportDiv.className = "report-item";
          reportDiv.innerHTML = `
                        <div class="report-meta">${report.timestamp}</div>
                        <div><span class="report-status ${report.statusClass}">${report.status}</span></div>
                        <div>${report.details}</div>
                    `;
          reportsContent.appendChild(reportDiv);
        });
      } else {
        reportsContent.innerHTML = "<p>Нет доступных отчетов.</p>";
      }
      showOverlay(reportsListPane);
    });
  } else {
    console.error("Element with ID 'reports-bell' not found!");
  }
  document.body.addEventListener("click", (event) => {
    /* ... (keep as is) ... */
    const overlay = event.target.closest(".details-overlay");
    if (!overlay) return;
    const backButton = event.target.closest(".back-button");
    const closeButton = event.target.closest(".close-button");
    if (backButton || closeButton) {
      hideOverlay(overlay);
    }
  });

  // --- Camera Feed Toggle Button Logic ---
  if (toggleFeedBtn && normalFeedImg) {
    toggleFeedBtn.addEventListener("click", () => {
      isLiveFeedActive = !isLiveFeedActive;
      console.log(
        `[Toggle] Switched mode. isLiveFeedActive: ${isLiveFeedActive}`,
      );

      if (normalFeedImg) {
        normalFeedImg.src = "";
        if (currentVideoUrl) {
          URL.revokeObjectURL(currentVideoUrl);
          currentVideoUrl = null;
        }
      }

      const newStream = isLiveFeedActive ? "normal_video" : "static_video";
      const oldStream = isLiveFeedActive ? "static_video" : "normal_video";
      // <<< MODIFIED Alt Text >>>
      const statusMsg = isLiveFeedActive
        ? "Запрос live потока (детектор движения)..."
        : "Запрос video потока (детектор движения)...";
      const buttonText = isLiveFeedActive ? "Показать Видео" : "Камера (Live)";

      toggleFeedBtn.textContent = buttonText;
      setVideoFeedStatus(statusMsg);

      if (activeTabId === "camera-pane") {
        sendWebSocketMessage({ action: "unsubscribe", stream: oldStream });
        sendWebSocketMessage({ action: "subscribe", stream: newStream });
        if (!socket || socket.readyState !== WebSocket.OPEN) {
          connectWebSocket();
        }
      } else {
        setVideoFeedStatus("");
      }
    });
  } else {
    console.error("Could not find toggle button or feed image element!");
  }

  // --- Initialisation ---
  console.log("DOM Loaded. Initializing...");
  if (toggleFeedBtn) {
    toggleFeedBtn.textContent = isLiveFeedActive
      ? "Показать Видео"
      : "Камера (Live)";
  }
  connectWebSocket(); // Initial connection attempt
}); // End DOMContentLoaded
// --- END OF FILE script.js ---
