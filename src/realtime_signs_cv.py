import cv2, numpy as np, mediapipe as mp, tensorflow as tf
from collections import deque
from pathlib import Path
import json, time

# --- paths & knobs ---
MODEL_PATH = r"C:\Users\sayu\Desktop\Sign_Language\RealTimeObjectDetection\models\sign_digits_classifier_old.keras"
IMG_SIZE   = (128, 128)
CONF_MIN   = 0.80          # stricter decision threshold
SMOOTH_N   = 11            # temporal smoothing length
MIRROR     = True

# --- load model ---
mpath = Path(MODEL_PATH)
if not mpath.exists():
    raise SystemExit(f"Model not found: {mpath}")
model = tf.keras.models.load_model(mpath)

# detect if model already rescales 0..255 -> 0..1 internally
first_layer_name = model.layers[0].__class__.__name__.lower()
HAS_RESCALING = "rescaling" in first_layer_name
print("Rescaling in model first layer:", HAS_RESCALING)

# --- label map (friendly names) ---
idx_to_name = {}
label_map_path = mpath.with_name("label_map.json")
if label_map_path.exists():
    try:
        with open(label_map_path, "r", encoding="utf-8") as f:
            raw = json.load(f)  # {"0":"zero","1":"one",...}
        idx_to_name = {int(k): v for k, v in raw.items()}
        print("Loaded label map with", len(idx_to_name), "classes")
    except Exception as e:
        print("Warning: failed to load label_map.json:", e)

# --- mediapipe setup ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)

def make_square_bbox(x1, y1, x2, y2, W, H, extra=0.08):  # ↓ smaller margin than before (was 0.20)
    cx, cy = (x1+x2)/2, (y1+y2)/2
    w, h = (x2-x1), (y2-y1)
    side = int(max(w, h) * (1.0 + extra))
    x1s = int(max(0, cx - side/2)); y1s = int(max(0, cy - side/2))
    x2s = int(min(W-1, x1s + side)); y2s = int(min(H-1, y1s + side))
    if x2s - x1s < side: x1s = max(0, x2s - side)
    if y2s - y1s < side: y1s = max(0, y2s - side)
    return x1s, y1s, x2s, y2s

def letterbox(img, size):
    tw, th = size
    h, w = img.shape[:2]
    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh))
    canvas = np.zeros((th, tw, 3), dtype=img.dtype)
    top  = (th - nh) // 2
    left = (tw - nw) // 2
    canvas[top:top+nh, left:left+nw] = resized
    return canvas

def lm_to_bbox(lms, margin=0.15):
    xs=[lm.x for lm in lms]; ys=[lm.y for lm in lms]
    x1,x2=max(0.0,min(xs)),min(1.0,max(xs))
    y1,y2=max(0.0,min(ys)),min(1.0,max(ys))
    mx=margin*(x2-x1+1e-6); my=margin*(y2-y1+1e-6)
    return max(0.0,x1-mx),max(0.0,y1-my),min(1.0,x2+mx),min(1.0,y2+my)

def open_cam():
    for backend in (cv2.CAP_DSHOW, cv2.CAP_ANY):
        for cam_id in (0,1,2):
            cap = cv2.VideoCapture(cam_id, backend)
            if cap.isOpened(): return cap
            cap.release()
    return None

cap = open_cam()
if cap is None:
    raise SystemExit("No webcam found (tried 0/1/2). Close other apps and try again.")

vote = deque(maxlen=SMOOTH_N); prev_t=time.time(); fps=0.0
print("Press 'q' or ESC to quit")

while True:
    ok, frame = cap.read()
    if not ok: break
    if MIRROR: frame = cv2.flip(frame, 1)

    h, w = frame.shape[:2]
    res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if res.multi_hand_landmarks:
        lms = res.multi_hand_landmarks[0].landmark
        x1n, y1n, x2n, y2n = lm_to_bbox(lms, margin=0.15)
        x1, y1, x2, y2 = int(x1n*w), int(y1n*h), int(x2n*w), int(y2n*h)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w-1, x2), min(h-1, y2)

        crop = frame[y1:y2, x1:x2]
        if crop.size and (x2-x1) > 10 and (y2-y1) > 10:
            # 1) square bbox + (smaller) margin
            x1s, y1s, x2s, y2s = make_square_bbox(x1, y1, x2, y2, w, h, extra=0.08)
            crop_sq = frame[y1s:y2s, x1s:x2s]

            # guard: skip tiny crops
            if (x2s - x1s) < 80 or (y2s - y1s) < 80:
                cv2.putText(frame, "Move hand closer", (10,50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                cv2.rectangle(frame, (x1s, y1s), (x2s, y2s), (0,0,255), 2)
                cv2.imshow("Sign Digit (0–9) — realtime (no matplotlib)", frame)
                if cv2.waitKey(1) & 0xFF in (27, ord('q')): break
                continue

            # 2) BGR -> RGB
            rgb = cv2.cvtColor(crop_sq, cv2.COLOR_BGR2RGB)

            # 3) letterbox to model size
            inp = letterbox(rgb, IMG_SIZE)

            # 4) light normalization (helps lighting/exposure)
            inp = cv2.GaussianBlur(inp, (3,3), 0)
            inp = cv2.normalize(inp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # 5) debug: see exactly what the model sees
            cv2.imshow("MODEL_INPUT_128x128", cv2.cvtColor(inp, cv2.COLOR_RGB2BGR))

            # 6) tensor prep (respect model rescaling)
            x = inp.astype(np.float32)[None, ...]
            if not HAS_RESCALING:
                x = x / 255.0

            p = model.predict(x, verbose=0)[0]
            cls = int(np.argmax(p)); conf = float(p[cls])

            # draw the square box
            cv2.rectangle(frame, (x1s, y1s), (x2s, y2s), (0,255,0), 2)

            if conf >= CONF_MIN:
                vote.append(cls)
                maj = max(set(vote), key=vote.count)
                label_name = idx_to_name.get(maj, str(maj))
                label_main = f"{label_name}  {conf:.2f}"
            else:
                vote.clear()  # avoid stale decisions when confidence is low
                label_main = "?"

            # top-3 display
            top3_idx = np.argsort(p)[-3:][::-1]
            top3 = " ".join([f"{idx_to_name.get(i, str(i))}:{p[i]:.2f}" for i in top3_idx])

            cv2.putText(frame, label_main, (x1s, max(20, y1s-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.putText(frame, top3, (x1s, y2s + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    # FPS overlay
    now = time.time()
    fps = fps*0.9 + (1.0/max(1e-6, now-prev_t))*0.1
    prev_t = now
    cv2.putText(frame, f"FPS: {fps:.1f}", (10,25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow("Sign Digit (0–9) — realtime (no matplotlib)", frame)
    if cv2.waitKey(1) & 0xFF in (27, ord('q')):
        break

cap.release(); cv2.destroyAllWindows(); hands.close()
