"""
services/webapp/app.py — Event audio webapp + manual reminder portal.

Port 8502. Two tabs:
  /  →  Live Events (auto-announces face recognition events via PiP audio)
  /  →  Reminders  (manually trigger water/food/medication/custom reminders)
"""
from __future__ import annotations

import asyncio
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, Response, JSONResponse, StreamingResponse

load_dotenv()

TEMPFILES = Path("/Users/mtb/Programming/Hackabull-2026/tempfiles")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "")
PATIENT_NAME = os.getenv("PATIENT_NAME", "there")

app = FastAPI()

# Per-client SSE queues — every connected browser gets its own queue.
# _broadcast() pushes to all of them simultaneously.
_clients: set[asyncio.Queue] = set()


def _broadcast(event: dict) -> None:
    """Push one event to every connected SSE client right now."""
    dead = set()
    for q in _clients:
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            dead.add(q)
    _clients.difference_update(dead)


@app.get("/stream")
async def stream():
    """SSE endpoint — each browser holds one open connection here."""
    q: asyncio.Queue = asyncio.Queue(maxsize=50)
    _clients.add(q)

    async def generator():
        try:
            while True:
                try:
                    event = await asyncio.wait_for(q.get(), timeout=20)
                    yield f"data: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"   # prevent proxy/load-balancer timeout
        finally:
            _clients.discard(q)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/ingest")
async def ingest(request: Request):
    event = await request.json()
    _broadcast(event)
    return {"status": "ok"}


@app.post("/remind")
async def remind(request: Request):
    body = await request.json()
    text = body.get("text", "").strip()
    if not text:
        return JSONResponse({"error": "no text"}, status_code=400)
    _broadcast({
        "event_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "type": "reminder",
        "subtype": "manual_reminder",
        "voice_script": text,
    })
    return {"status": "ok"}


DETECTION_PAUSED_FLAG = Path("/Users/mtb/Programming/Hackabull-2026/tempfiles/detection_paused")


@app.post("/control")
async def control(request: Request):
    """Broadcast a control command to every connected client and apply server-side effects."""
    body = await request.json()
    command = body.get("command", "")

    if command == "pause_detection":
        DETECTION_PAUSED_FLAG.parent.mkdir(exist_ok=True)
        DETECTION_PAUSED_FLAG.touch()
    elif command == "resume_detection":
        DETECTION_PAUSED_FLAG.unlink(missing_ok=True)

    _broadcast({"type": "control", "command": command})
    return {"status": "ok"}


async def _tts_elevenlabs(text: str) -> bytes:
    from elevenlabs.client import ElevenLabs
    client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
    audio_iter = client.text_to_speech.convert(
        voice_id=ELEVENLABS_VOICE_ID,
        text=text,
        model_id="eleven_flash_v2_5",
    )
    return b"".join(audio_iter)


async def _tts_edge(text: str) -> bytes:
    import edge_tts
    chunks: list[bytes] = []
    communicate = edge_tts.Communicate(text, voice="en-US-AriaNeural")
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            chunks.append(chunk["data"])
    return b"".join(chunks)


@app.post("/speak")
async def speak(request: Request):
    body = await request.json()
    text = body.get("text", "").strip()
    if not text:
        return JSONResponse({"error": "no text"}, status_code=400)

    if ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID:
        try:
            audio_bytes = await _tts_elevenlabs(text)
            return Response(content=audio_bytes, media_type="audio/mpeg")
        except Exception:
            pass

    try:
        audio_bytes = await _tts_edge(text)
        return Response(content=audio_bytes, media_type="audio/mpeg")
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.get("/", response_class=HTMLResponse)
def index():
    name = PATIENT_NAME
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Nazr</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #0f1117;
      color: #e0e0e0;
    }}

    /* ── Top bar ── */
    .topbar {{
      background: #13151f;
      border-bottom: 1px solid #2a2d3a;
      padding: 14px 24px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 10px;
    }}
    .topbar h1 {{ font-size: 1.25rem; color: #fff; }}
    .topbar-right {{ display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }}
    #status {{ font-size: 0.82rem; color: #888; }}
    #status.speaking {{ color: #4ade80; }}

    /* ── Tab bar ── */
    .tabs {{
      display: flex;
      border-bottom: 1px solid #2a2d3a;
      background: #13151f;
    }}
    .tab-btn {{
      flex: 1;
      padding: 12px 0;
      background: none;
      border: none;
      border-bottom: 3px solid transparent;
      color: #888;
      font-size: 0.95rem;
      cursor: pointer;
      transition: color 0.15s, border-color 0.15s;
    }}
    .tab-btn:hover {{ color: #ccc; }}
    .tab-btn.active {{ color: #fff; border-bottom-color: #60a5fa; }}

    /* ── Shared controls ── */
    .audio-bar {{
      display: flex; gap: 10px; padding: 14px 24px;
      background: #0f1117; border-bottom: 1px solid #1e2130;
      flex-wrap: wrap; align-items: center;
    }}
    .btn {{
      background: #1a1d27;
      border: 1px solid #2a2d3a;
      color: #e0e0e0;
      padding: 9px 16px;
      border-radius: 8px;
      cursor: pointer;
      font-size: 0.88rem;
      white-space: nowrap;
    }}
    .btn:hover {{ background: #2a2d3a; }}
    .btn:disabled {{ opacity: 0.4; cursor: default; }}
    .btn.pip-active {{ border-color: #4ade80; color: #4ade80; }}

    /* ── Panel containers ── */
    .panel {{ display: none; padding: 24px; max-width: 780px; margin: 0 auto; }}
    .panel.active {{ display: block; }}

    /* ── Live events panel ── */
    #feed {{ display: flex; flex-direction: column; gap: 14px; }}
    .card {{
      background: #1a1d27; border: 1px solid #2a2d3a;
      border-radius: 10px; padding: 18px 22px;
      animation: slideIn 0.25s ease;
    }}
    .card.identity {{ border-left: 4px solid #60a5fa; }}
    .card.health   {{ border-left: 4px solid #4ade80; }}
    .card.reminder {{ border-left: 4px solid #f59e0b; }}
    @keyframes slideIn {{
      from {{ opacity: 0; transform: translateY(-6px); }}
      to   {{ opacity: 1; transform: translateY(0); }}
    }}
    .card-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }}
    .badge {{ font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; padding: 3px 9px; border-radius: 20px; }}
    .badge.identity {{ background: #1e3a5f; color: #60a5fa; }}
    .badge.health   {{ background: #14532d; color: #4ade80; }}
    .badge.reminder {{ background: #451a03; color: #f59e0b; }}
    .ts {{ font-size: 0.75rem; color: #555; }}
    .voice-script {{ font-size: 1rem; line-height: 1.5; color: #d1d5db; margin-bottom: 6px; }}
    .meta {{ font-size: 0.78rem; color: #6b7280; }}

    /* ── Reminders panel ── */
    .reminder-section {{ margin-bottom: 28px; }}
    .reminder-section h2 {{
      font-size: 0.8rem; font-weight: 600; letter-spacing: 0.1em;
      text-transform: uppercase; color: #555; margin-bottom: 12px;
    }}
    .reminder-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
      gap: 10px;
    }}
    .r-btn {{
      background: #1a1d27;
      border: 1px solid #2a2d3a;
      border-radius: 10px;
      padding: 14px 16px;
      color: #e0e0e0;
      font-size: 0.92rem;
      text-align: left;
      cursor: pointer;
      line-height: 1.4;
      transition: background 0.15s, border-color 0.15s;
    }}
    .r-btn:hover {{ background: #22263a; border-color: #3a3d50; }}
    .r-btn:active {{ background: #2a2d45; }}
    .r-btn .r-icon {{ font-size: 1.3rem; display: block; margin-bottom: 6px; }}
    .r-btn.water:hover   {{ border-color: #38bdf8; }}
    .r-btn.food:hover    {{ border-color: #4ade80; }}
    .r-btn.meds:hover    {{ border-color: #c084fc; }}
    .r-btn.checkin:hover {{ border-color: #f59e0b; }}

    .custom-box {{
      background: #1a1d27;
      border: 1px solid #2a2d3a;
      border-radius: 10px;
      padding: 18px;
    }}
    .custom-box textarea {{
      width: 100%; min-height: 80px;
      background: #111318; border: 1px solid #2a2d3a;
      border-radius: 8px; color: #e0e0e0;
      font-size: 0.95rem; padding: 12px;
      resize: vertical; font-family: inherit;
    }}
    .custom-box textarea:focus {{ outline: none; border-color: #60a5fa; }}
    .custom-send {{
      margin-top: 10px;
      background: #1e3a5f; border: 1px solid #2563eb;
      color: #93c5fd; padding: 10px 20px;
      border-radius: 8px; cursor: pointer; font-size: 0.9rem;
    }}
    .custom-send:hover {{ background: #1e40af; }}

    .sent-log {{
      margin-top: 16px;
      font-size: 0.8rem;
      color: #4ade80;
      min-height: 20px;
    }}

    /* ── Hidden PiP elements ── */
    #pip-video {{ position: absolute; width: 1px; height: 1px; opacity: 0; pointer-events: none; }}
    #pip-canvas {{ display: none; }}
  </style>
</head>
<body>

  <!-- Top bar -->
  <div class="topbar">
    <h1>Nazr</h1>
    <div class="topbar-right">
      <span id="status">Tap Start to activate audio</span>
    </div>
  </div>

  <!-- Audio controls (shared across tabs) -->
  <div class="audio-bar">
    <button class="btn" id="start-btn" onclick="startAudio()">▶ Start</button>
    <button class="btn" id="pip-btn"   onclick="togglePiP()" disabled>⧉ PiP</button>
    <button class="btn" id="mute-btn"  onclick="toggleMute()" disabled>🔊 Mute</button>
    <button class="btn" id="muteall-btn" onclick="muteAll()" style="border-color:#ef4444;color:#ef4444;">🔇 Silence Everyone</button>
    <button class="btn" id="detection-btn" onclick="toggleDetection()" style="border-color:#4ade80;color:#4ade80;">🔍 Detection: ON</button>
    <span style="font-size:0.78rem;color:#555;">Enable PiP then switch to SpecBridge — audio plays through glasses.</span>
  </div>

  <!-- Tabs -->
  <div class="tabs">
    <button class="tab-btn active" id="tab-events-btn"    onclick="showTab('events')">📡 Live Events</button>
    <button class="tab-btn"        id="tab-reminders-btn" onclick="showTab('reminders')">🔔 Reminders</button>
  </div>

  <!-- Hidden PiP elements -->
  <canvas id="pip-canvas" width="320" height="180"></canvas>
  <video  id="pip-video" playsinline></video>

  <!-- ── Live Events panel ── -->
  <div class="panel active" id="panel-events">
    <div id="feed"></div>
  </div>

  <!-- ── Reminders panel ── -->
  <div class="panel" id="panel-reminders">

    <div class="reminder-section">
      <h2>👤 Announce person</h2>
      <div class="reminder-grid">
        <button class="r-btn" style="border-color:#4ade80;"
          onclick="sendReminder(this, '{name}, your son Mohammed is here. He is a student at USF. Last time you spoke, you were talking about how he has never won a game in basketball.')">
          <span class="r-icon">🟢</span><strong>Mohammed</strong><br><span style="font-size:0.8rem;color:#9ca3af;">son · green shirt</span>
        </button>
        <button class="r-btn" style="border-color:#94a3b8;"
          onclick="sendReminder(this, '{name}, your grandson Taikhoom is here. He is a chemical engineering student at USF. Last time you spoke, you were talking about his next camping trip.')">
          <span class="r-icon">⚫</span><strong>Taikhoom</strong><br><span style="font-size:0.8rem;color:#9ca3af;">grandson · black shirt</span>
        </button>
        <button class="r-btn" style="border-color:#fbbf24;"
          onclick="sendReminder(this, '{name}, your neighbor Ismail is here. He is a chemical engineer from Intel. Last time you spoke, you were talking about the new Bollywood movie in theater.')">
          <span class="r-icon">🟡</span><strong>Ismail</strong><br><span style="font-size:0.8rem;color:#9ca3af;">neighbor · yellow shirt</span>
        </button>
      </div>
    </div>

    <div class="reminder-section">
      <h2>💧 Water</h2>
      <div class="reminder-grid">
        <button class="r-btn water" onclick="sendReminder(this, '{name}, you have not had any water in over 8 hours. Please drink a glass of water right now.')">
          <span class="r-icon">💧</span>No water in 8 hours
        </button>
        <button class="r-btn water" onclick="sendReminder(this, '{name}, please remember to drink some water. Staying hydrated is very important.')">
          <span class="r-icon">🥤</span>Hydration reminder
        </button>
      </div>
    </div>

    <div class="reminder-section">
      <h2>🍽️ Food</h2>
      <div class="reminder-grid">
        <button class="r-btn food" onclick="sendReminder(this, '{name}, you have not eaten in several hours. It is time for a meal. Please come eat something.')">
          <span class="r-icon">🍽️</span>Hasn't eaten in hours
        </button>
        <button class="r-btn food" onclick="sendReminder(this, '{name}, it is lunch time. Please come have something to eat.')">
          <span class="r-icon">🥗</span>Lunch time
        </button>
        <button class="r-btn food" onclick="sendReminder(this, '{name}, dinner is ready. Please come to the table.')">
          <span class="r-icon">🍲</span>Dinner time
        </button>
      </div>
    </div>

    <div class="reminder-section">
      <h2>💊 Medication</h2>
      <div class="reminder-grid">
        <button class="r-btn meds" onclick="sendReminder(this, '{name}, it is time to take your morning medication. Please do not forget.')">
          <span class="r-icon">💊</span>Morning medication
        </button>
        <button class="r-btn meds" onclick="sendReminder(this, '{name}, it is time to take your evening medication.')">
          <span class="r-icon">🌙</span>Evening medication
        </button>
      </div>
    </div>

    <div class="reminder-section">
      <h2>👋 Check-in</h2>
      <div class="reminder-grid">
        <button class="r-btn checkin" onclick="sendReminder(this, 'Good morning, {name}! How are you feeling today?')">
          <span class="r-icon">🌅</span>Good morning
        </button>
        <button class="r-btn checkin" onclick="sendReminder(this, '{name}, just checking in. Is there anything you need right now?')">
          <span class="r-icon">❤️</span>Just checking in
        </button>
        <button class="r-btn checkin" onclick="sendReminder(this, '{name}, do not forget to take a short walk today. It is good for you.')">
          <span class="r-icon">🚶</span>Go for a walk
        </button>
      </div>
    </div>

    <div class="reminder-section">
      <h2>✏️ Custom message</h2>
      <div class="custom-box">
        <textarea id="custom-text" placeholder="Type any reminder or message to speak through the glasses…"></textarea>
        <button class="custom-send" onclick="sendCustom()">▶ Send to glasses</button>
      </div>
      <div class="sent-log" id="sent-log"></div>
    </div>

  </div><!-- /panel-reminders -->

  <script>
    const PATIENT = "{name}";
    let muted = false;
    let audioCtx = null;
    let audioDest = null;
    let pipVideo = null;
    let canvasCtx = null;
    let pipActive = false;
    let statusText = 'Listening for events…';
    let lastEvent = null;

    // ── Tab switching ──────────────────────────────────────────────────────────
    function showTab(name) {{
      document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.getElementById('panel-' + name).classList.add('active');
      document.getElementById('tab-' + name + '-btn').classList.add('active');
    }}

    // ── Canvas drawing (what appears in the PiP window) ───────────────────────
    function drawPiP() {{
      if (!canvasCtx) return;
      const c = document.getElementById('pip-canvas');
      const w = c.width, h = c.height;
      canvasCtx.fillStyle = '#0f1117';
      canvasCtx.fillRect(0, 0, w, h);
      canvasCtx.fillStyle = '#1a1d27';
      canvasCtx.fillRect(0, 0, w, 36);
      canvasCtx.fillStyle = '#4ade80';
      canvasCtx.font = 'bold 15px sans-serif';
      canvasCtx.textAlign = 'left';
      canvasCtx.fillText('Nazr', 12, 24);
      if (lastEvent) {{
        const ev = lastEvent;
        const isIdentity = ev.type === 'identity';
        canvasCtx.fillStyle = isIdentity ? '#60a5fa' : ev.type === 'reminder' ? '#f59e0b' : '#4ade80';
        canvasCtx.font = 'bold 12px sans-serif';
        const badge = (ev.subtype || ev.type || '').replace(/_/g,' ').toUpperCase();
        canvasCtx.fillText(badge, 12, 58);
        const script = ev.voice_script || '';
        canvasCtx.fillStyle = '#d1d5db';
        canvasCtx.font = '12px sans-serif';
        wrapText(canvasCtx, script, 12, 78, w - 24, 17);
      }} else {{
        canvasCtx.fillStyle = '#555';
        canvasCtx.font = '13px sans-serif';
        canvasCtx.textAlign = 'center';
        canvasCtx.fillText(statusText, w / 2, h / 2);
      }}
      requestAnimationFrame(drawPiP);
    }}

    function wrapText(ctx, text, x, y, maxWidth, lineHeight) {{
      const words = text.split(' ');
      let line = '';
      for (const word of words) {{
        const test = line ? line + ' ' + word : word;
        if (ctx.measureText(test).width > maxWidth && line) {{
          ctx.fillText(line, x, y);
          line = word; y += lineHeight;
          if (y > 168) {{ ctx.fillText(line + '…', x, y); return; }}
        }} else {{ line = test; }}
      }}
      if (line) ctx.fillText(line, x, y);
    }}

    // ── Audio setup ────────────────────────────────────────────────────────────
    function startAudio() {{
      if (audioCtx) return;
      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      audioDest = audioCtx.createMediaStreamDestination();

      // iOS suspends AudioContext when the app goes to PiP/background, which
      // causes audio.start() calls to queue and play all at once when foregrounded.
      // A silent 1Hz oscillator keeps a continuous signal flowing through the
      // AudioContext so iOS sees the audio session as active and never suspends it.
      const kaGain = audioCtx.createGain();
      kaGain.gain.value = 0.001;           // -60 dB — completely inaudible
      const kaOsc = audioCtx.createOscillator();
      kaOsc.frequency.value = 1;           // 1 Hz — below human hearing
      kaOsc.connect(kaGain);
      kaGain.connect(audioDest);           // into PiP stream only, not speakers
      kaOsc.start();

      // Belt-and-suspenders: auto-resume if iOS still suspends the context
      audioCtx.onstatechange = () => {{
        if (audioCtx.state === 'suspended') audioCtx.resume().catch(() => {{}});
      }};

      const canvas = document.getElementById('pip-canvas');
      canvasCtx = canvas.getContext('2d');
      drawPiP();
      const videoStream = canvas.captureStream(10);
      const audioTrack = audioDest.stream.getAudioTracks()[0];
      if (audioTrack) videoStream.addTrack(audioTrack);
      pipVideo = document.getElementById('pip-video');
      pipVideo.srcObject = videoStream;
      pipVideo.muted = false;
      pipVideo.play().catch(() => {{}});
      document.getElementById('start-btn').textContent = '✓ Audio Ready';
      document.getElementById('start-btn').disabled = true;
      document.getElementById('pip-btn').disabled = false;
      document.getElementById('mute-btn').disabled = false;
      document.getElementById('status').textContent = 'Listening for events…';
      pipVideo.addEventListener('leavepictureinpicture', () => {{
        pipActive = false;
        document.getElementById('pip-btn').textContent = '⧉ PiP';
        document.getElementById('pip-btn').classList.remove('pip-active');
      }});
    }}

    // ── PiP toggle ─────────────────────────────────────────────────────────────
    async function togglePiP() {{
      if (!audioCtx) {{ startAudio(); return; }}
      if (audioCtx.state === 'suspended') await audioCtx.resume();
      try {{
        if (!document.pictureInPictureElement) {{
          await pipVideo.requestPictureInPicture();
          pipActive = true;
          document.getElementById('pip-btn').textContent = '⧉ Exit PiP';
          document.getElementById('pip-btn').classList.add('pip-active');
        }} else {{
          await document.exitPictureInPicture();
        }}
      }} catch (e) {{ alert('PiP error: ' + e.message); }}
    }}

    // ── Mute ───────────────────────────────────────────────────────────────────
    function toggleMute() {{
      muted = !muted;
      document.getElementById('mute-btn').textContent = muted ? '🔇 Unmute' : '🔊 Mute';
    }}

    function muteAll() {{
      fetch('/control', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{command: 'mute_all'}}),
      }});
    }}

    let detectionOn = true;
    function toggleDetection() {{
      const cmd = detectionOn ? 'pause_detection' : 'resume_detection';
      fetch('/control', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{command: cmd}}),
      }});
    }}

    function applyDetectionState(on) {{
      detectionOn = on;
      const btn = document.getElementById('detection-btn');
      if (on) {{
        btn.textContent = '🔍 Detection: ON';
        btn.style.borderColor = '#4ade80'; btn.style.color = '#4ade80';
      }} else {{
        btn.textContent = '⏸ Detection: OFF (manual)';
        btn.style.borderColor = '#f59e0b'; btn.style.color = '#f59e0b';
      }}
    }}

    // ── Audio playback ─────────────────────────────────────────────────────────
    function speakFallback(text) {{
      const synth = window.speechSynthesis; synth.cancel();
      const utt = new SpeechSynthesisUtterance(text); utt.rate = 0.95;
      synth.speak(utt);
    }}

    async function speak(text) {{
      if (muted || !text) return;
      const statusEl = document.getElementById('status');

      try {{
        const res = await fetch('/speak', {{
          method: 'POST',
          headers: {{'Content-Type': 'application/json'}},
          body: JSON.stringify({{text}}),
        }});
        if (!res.ok) {{ speakFallback(text); return; }}

        const bytes = await res.arrayBuffer();

        // ── Speakers: plain Audio element — reliable on every browser/OS ──
        const blobUrl = URL.createObjectURL(new Blob([bytes], {{type: 'audio/mpeg'}}));
        const audioEl = new Audio(blobUrl);
        audioEl.onended = () => {{
          URL.revokeObjectURL(blobUrl);
          statusText = 'Listening for events…';
          statusEl.textContent = 'Listening for events…'; statusEl.className = '';
        }};
        statusText = '🔊 Speaking…';
        statusEl.textContent = '🔊 Speaking…'; statusEl.className = 'speaking';
        audioEl.play().catch(() => speakFallback(text));

        // ── PiP / glasses: route through AudioContext → MediaStreamDestination ──
        if (audioCtx && audioDest) {{
          if (audioCtx.state === 'suspended') await audioCtx.resume();
          try {{
            const buf = await audioCtx.decodeAudioData(bytes.slice(0));
            const src = audioCtx.createBufferSource();
            src.buffer = buf;
            src.connect(audioDest);
            src.start();
          }} catch (_) {{}}  // PiP routing failing never breaks speaker playback
        }}
      }} catch (_) {{ speakFallback(text); }}
    }}

    // ── Reminder helpers ───────────────────────────────────────────────────────
    // POST to /remind → server pushes into the shared poll queue →
    // EVERY connected client picks it up and plays it (broadcast).
    async function sendReminder(btn, text) {{
      if (!audioCtx) startAudio();
      btn.disabled = true;
      btn.style.borderColor = '#f59e0b';
      try {{
        await fetch('/remind', {{
          method: 'POST',
          headers: {{'Content-Type': 'application/json'}},
          body: JSON.stringify({{text}}),
        }});
        showSentLog(text);
      }} catch(e) {{ showSentLog('Error: ' + e.message); }}
      setTimeout(() => {{ btn.disabled = false; btn.style.borderColor = ''; }}, 2000);
    }}

    async function sendCustom() {{
      const ta = document.getElementById('custom-text');
      const text = ta.value.trim();
      if (!text) return;
      if (!audioCtx) startAudio();
      try {{
        await fetch('/remind', {{
          method: 'POST',
          headers: {{'Content-Type': 'application/json'}},
          body: JSON.stringify({{text}}),
        }});
        showSentLog(text);
        ta.value = '';
      }} catch(e) {{ showSentLog('Error: ' + e.message); }}
    }}

    function showSentLog(text) {{
      const log = document.getElementById('sent-log');
      const short = text.length > 80 ? text.slice(0, 80) + '…' : text;
      log.textContent = '✓ Sent: ' + short;
      setTimeout(() => {{ log.textContent = ''; }}, 5000);
    }}

    // ── Live events cards ──────────────────────────────────────────────────────
    function voiceText(ev) {{
      if (ev.voice_script) return ev.voice_script;
      if (ev.type === 'health') return 'Health alert: ' + (ev.metadata?.detected_item || ev.subtype) + ' detected.';
      return (ev.metadata?.person_profile?.name || 'someone') + ' has been detected.';
    }}

    function formatTs(ts) {{
      try {{ return new Date(ts).toLocaleTimeString(); }} catch {{ return ts; }}
    }}

    function addCard(ev) {{
      lastEvent = ev;
      const feed = document.getElementById('feed');
      const card = document.createElement('div');
      card.className = 'card ' + (ev.type || '');
      const subtype = (ev.subtype || '').replace(/_/g, ' ');
      let body = '';
      if (ev.voice_script) body += `<div class="voice-script">${{ev.voice_script}}</div>`;
      if (ev.type === 'health') body += `<div class="meta">Detected: <strong>${{ev.metadata?.detected_item || '—'}}</strong></div>`;
      if (ev.type === 'identity') {{
        const p = ev.metadata?.person_profile || {{}};
        if (p.name) body += `<div class="meta">${{p.relationship || ''}} · ${{p.name}}</div>`;
      }}
      card.innerHTML = `<div class="card-header"><span class="badge ${{ev.type}}">${{subtype || ev.type}}</span><span class="ts">${{formatTs(ev.timestamp || '')}}</span></div>${{body}}`;
      feed.insertBefore(card, feed.firstChild);
    }}

    // ── SSE — one persistent connection, server pushes to ALL clients ─────────
    const evtSource = new EventSource('/stream');
    evtSource.onmessage = async (e) => {{
      try {{
        const ev = JSON.parse(e.data);
        if (ev.type === 'control') {{
          if (ev.command === 'mute_all') {{
            muted = true;
            document.getElementById('mute-btn').textContent = '🔇 Unmute';
            document.getElementById('muteall-btn').textContent = '🔇 Silenced';
            setTimeout(() => {{ document.getElementById('muteall-btn').textContent = '🔇 Silence Everyone'; }}, 3000);
          }} else if (ev.command === 'pause_detection') {{
            applyDetectionState(false);
          }} else if (ev.command === 'resume_detection') {{
            applyDetectionState(true);
          }}
          return;
        }}
        addCard(ev);
        await speak(voiceText(ev));
      }} catch (_) {{}}
    }};
    // EventSource reconnects automatically on drop — no extra logic needed.
  </script>
</body>
</html>"""
