const BACKEND_URL = "http://127.0.0.1:5000/convert";

async function callBackend(text) {
  const res = await fetch(BACKEND_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text })
  });
  if (!res.ok) throw new Error("Backend error");
  return res.json();
}

function playUrlsSequentially(urls) {
  const video = document.getElementById("signVideo");
  if (!video) return;

  const queue = (urls || []).filter(u => typeof u === "string" && u.trim().length > 0);

  if (queue.length === 0) return;

  video.muted = true;
  video.playsInline = true;

  let i = 0;

  function playNext() {
    if (i >= queue.length) return;

    const url = queue[i];
    i++;

    video.onended = playNext;
    video.onerror = playNext;   // ✅ if one video fails, skip to next

    video.src = url;
    video.load();

    video.play().catch(() => {
      // If autoplay still blocks, user can press play once (but proxy + muted usually avoids this)
    });
  }

  playNext();
}

window.startSpeech = function () {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) { alert("Use Google Chrome"); return; }

  const rec = new SR();
  rec.lang = "en-US";
  rec.interimResults = false;

  rec.onresult = async (e) => {
    const text = e.results[0][0].transcript || "";
    document.getElementById("spoken").innerText = "Text: " + text;

    try {
      const data = await callBackend(text);
      document.getElementById("gloss").innerText = "Gloss: " + data.gloss;
      playUrlsSequentially(data.urls);
    } catch (err) {
      console.error(err);
    }
  };

  rec.start();
};
