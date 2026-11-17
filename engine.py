"""スプリント バイオメカニクス解析エンジン (Streamlit Community Cloud 対応版)

- MediaPipe Tasks API (PoseLandmarker) を使用 (mp.solutions は使わない)
- cv2 / mediapipe は関数内で遅延インポート (起動時のフリーズ防止)
- フレームは横幅 max_width(既定640px) に縮小して処理し、メモリを節約
- 最初に映った (またはプレビューで指定した) 走者「1人だけ」をロックオンして追跡する

追跡の仕組み (元の mp.solutions.pose の内部トラッカーに相当するものを自前で実装):
  1. 前フレームの骨格の周辺だけを切り出して検出する (ROI追跡)。画面全体を毎回見ないので、
     近くに別の走者が並んでも、そちらへ乗り移りにくい。
  2. 検出された候補は「位置・体の大きさ・服の色(上半身/太もも)」がロック中の人と合うものだけ採用する。
  3. 合う人がいなければ「見失い」。一定時間続くか、画面端で消えたら解析を終了し、二度と再取得しない。
"""
import gc
import math
import os
import shutil
import subprocess
import tempfile
import threading
import urllib.request

import numpy as np

MODEL_URLS = {
    "lite": ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"),
    "full": ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_full/float16/latest/pose_landmarker_full.task"),
}

POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
    (27, 29), (29, 31), (27, 31), (28, 30), (30, 32), (28, 32),
]

# 追跡の厳しさ: app_thr = 服の色の違いの許容上限(0=同じ〜1=全く違う), lost_sec = 見失ってから終了するまでの秒数
STRICTNESS = {
    "緩い": {"app_thr": 0.65, "lost_sec": 0.5},
    "標準": {"app_thr": 0.55, "lost_sec": 0.3},
    "厳しい": {"app_thr": 0.45, "lost_sec": 0.25},
}

_IMG_LMS = {}
_IMG_LOCK = threading.Lock()


# ----------------------------------------------------------------------
# ラルフ・マン基準の評価ロジック (従来どおり)
# ----------------------------------------------------------------------
def _xy(p):
    if hasattr(p, "x"):
        return p.x, p.y
    return float(p[0]), float(p[1])


def calculate_angle(a, b, c):
    ax, ay = _xy(a)
    bx, by = _xy(b)
    cx, cy = _xy(c)
    radians = math.atan2(cy - by, cx - bx) - math.atan2(ay - by, ax - bx)
    angle = abs(radians * 180.0 / math.pi)
    if angle > 180.0:
        angle = 360 - angle
    return int(angle)


def evaluate_knee(angle):
    if 152 <= angle <= 155:
        return "ELITE (Ideal Range)", "", ""
    elif angle < 152:
        return "WARNING (Too Bent)", "エキセントリック筋力不足 (潰れ)", "デプスジャンプ、アンクルホップ"
    else:
        return "WARNING (Over-Stride)", "ハムストリングス・臀筋の引き戻し力不足", "ストレートレッグラン、Aスキップ"


def evaluate_hip(angle):
    if 170 <= angle <= 176:
        return "ELITE (Ideal Range)", "", ""
    elif angle < 170:
        return "WARNING (Leaned/Core Lock)", "腹圧・コアの固定力不足 (腰落ち)", "ウォールピストン、動的プランク"
    else:
        return "WARNING (Upright)", "体幹前面のホールド力不足", "シザースもも上げ"


# ----------------------------------------------------------------------
# 動画ユーティリティ
# ----------------------------------------------------------------------
def get_video_info(path):
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(
            "動画を開けませんでした。iPhoneのHEVC(H.265)動画などは読み込めない場合があります。"
            "H.264のmp4に変換してお試しください。"
        )
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    if fps <= 0 or fps > 240:
        fps = 30
    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(fps),
        "frames": max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))),
    }
    cap.release()
    return info


def resize_frame(img, max_width=640):
    """横幅 max_width に縮小 (幅・高さは偶数に) し、連続メモリ配列にして返す。"""
    import cv2

    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / float(w)
        w, h = max_width, int(round(h * scale))
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    h2, w2 = img.shape[0] - img.shape[0] % 2, img.shape[1] - img.shape[1] % 2
    if (h2, w2) != img.shape[:2]:
        img = img[:h2, :w2]
    return np.ascontiguousarray(img)


def read_frame(path, idx, max_width=640):
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("動画を開けませんでした。")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
    ok, img = cap.read()
    cap.release()
    if not ok:
        return None
    return resize_frame(img, max_width)


def ensure_model(kind="full"):
    """モデル(.task)を /tmp に自動ダウンロード。full が取得できなければ lite にフォールバック。"""
    order = [kind, "lite"] if kind != "lite" else ["lite"]
    for k in order:
        path = os.path.join(tempfile.gettempdir(), f"pose_landmarker_{k}.task")
        if os.path.exists(path) and os.path.getsize(path) > 1_000_000:
            return path
        try:
            tmp = path + ".part"
            urllib.request.urlretrieve(MODEL_URLS[k], tmp)
            os.replace(tmp, path)
            return path
        except Exception:
            if k == "lite":
                raise


# ----------------------------------------------------------------------
# PoseLandmarker (Tasks API)
# ----------------------------------------------------------------------
def _make_landmarker(video_mode, num_poses, kind="full"):
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    mode = vision.RunningMode.VIDEO if video_mode else vision.RunningMode.IMAGE
    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=ensure_model(kind)),
        running_mode=mode,
        num_poses=num_poses,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.PoseLandmarker.create_from_options(options)


def _to_people(result):
    people = []
    for lms in result.pose_landmarks or []:
        arr = np.array(
            [[l.x, l.y, l.z, getattr(l, "visibility", 1.0) or 0.0] for l in lms],
            dtype=np.float32,
        )
        people.append(arr)
    return people


def _detect(landmarker, frame_bgr, ts_ms=None):
    import cv2
    import mediapipe as mp

    rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    if ts_ms is None:
        res = landmarker.detect(mp_img)
    else:
        res = landmarker.detect_for_video(mp_img, int(ts_ms))
    return _to_people(res)


def _get_img_landmarker(kind="full"):
    if kind not in _IMG_LMS:
        _IMG_LMS[kind] = _make_landmarker(video_mode=False, num_poses=5, kind=kind)
    return _IMG_LMS[kind]


def person_center(lms):
    idx = [11, 12, 23, 24]
    return float(lms[idx, 0].mean()), float(lms[idx, 1].mean())


def person_height(lms):
    ok = lms[:, 3] > 0.3
    if ok.sum() < 4:
        return 0.0
    ys = lms[ok, 1]
    return float(ys.max() - ys.min())


def person_quality(lms):
    return float(lms[[11, 12, 23, 24], 3].mean())


def lowest_ankle_y(lms):
    return float(max(lms[27, 1], lms[28, 1]))


def detect_people_image(frame_bgr, kind="full"):
    """プレビュー用の単一画像検出。左→右の順に並べて返す。"""
    with _IMG_LOCK:
        people = _detect(_get_img_landmarker(kind), frame_bgr)
    people = [p for p in people if person_quality(p) >= 0.3]
    people.sort(key=lambda p: person_center(p)[0])
    return people


def detect_people_roi(frame_bgr, box, kind="full", min_h=320):
    """画像の一部(box=x0,y0,x1,y1 ピクセル)だけを切り出して検出し、座標を元の画像基準(0〜1)に戻して返す。
    切り出しが小さい(遠くの走者)ときは min_h px まで拡大して検出精度を上げる。"""
    import cv2

    x0, y0, x1, y1 = box
    H, W = frame_bgr.shape[:2]
    crop = np.ascontiguousarray(frame_bgr[y0:y1, x0:x1])
    if crop.shape[0] < 16 or crop.shape[1] < 16:
        return []
    ch, cw = crop.shape[:2]
    crop_in = crop
    if ch < min_h:
        s = min_h / float(ch)
        crop_in = cv2.resize(crop, (max(1, int(cw * s)), min_h), interpolation=cv2.INTER_CUBIC)
    with _IMG_LOCK:
        people = _detect(_get_img_landmarker(kind), crop_in)
    out = []
    for p in people:
        q = p.copy()
        q[:, 0] = (x0 + p[:, 0] * cw) / W
        q[:, 1] = (y0 + p[:, 1] * ch) / H
        out.append(q)
    return out


def default_target_index(people):
    """プレビューの初期ターゲット = 最も大きく映っている人。"""
    if not people:
        return None
    return int(np.argmax([person_height(p) for p in people]))


def pick_nearest(people, point):
    if not people or point is None:
        return None
    d = [math.hypot(person_center(p)[0] - point[0], person_center(p)[1] - point[1]) for p in people]
    return int(np.argmin(d))


def match_center(people, center, max_d=0.2):
    """記録済みのターゲット中心に最も近い人の index（遠ければ None）。"""
    if not people or center is None:
        return None
    d = [math.hypot(person_center(p)[0] - center[0], person_center(p)[1] - center[1]) for p in people]
    i = int(np.argmin(d))
    return i if d[i] <= max_d else None


def find_first_person_frame(path, start=0, step=3, max_seconds=20, max_width=480, kind="full"):
    """動画の先頭から走査し、最初に人がはっきり映るフレーム番号を返す。"""
    import cv2

    info = get_video_info(path)
    limit = min(info["frames"], start + int(info["fps"] * max_seconds))
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    idx = start
    found = None
    while idx < limit:
        ok, img = cap.read()
        if not ok:
            break
        img = resize_frame(img, max_width)
        for p in detect_people_image(img, kind):
            if person_height(p) >= 0.15 and person_quality(p) >= 0.5:
                found = idx
                break
        if found is not None:
            break
        for _ in range(step - 1):
            cap.grab()
        idx += step
    cap.release()
    return found


# ----------------------------------------------------------------------
# 服装(見た目)による本人照合
# ----------------------------------------------------------------------
_HSV_BINS = [6, 3, 4]  # 色相6 x 彩度3 x 明度4 (白シャツと黒シャツを明度で区別するため明度も使う)


def _region_hist(hsv, x0, y0, x1, y1, core=0.55):
    """領域の横方向は中央 core 割だけを使う (隣の走者や背景の色が混ざるのを防ぐ)。"""
    import cv2

    H, W = hsv.shape[:2]
    cx, half = (x0 + x1) / 2.0, (x1 - x0) / 2.0 * core
    x0, x1 = cx - half, cx + half
    x0, x1 = max(0, int(x0)), min(W, int(math.ceil(x1)))
    y0, y1 = max(0, int(y0)), min(H, int(y1))
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None
    h = cv2.calcHist([hsv[y0:y1, x0:x1]], [0, 1, 2], None, _HSV_BINS,
                     [0, 180, 0, 256, 0, 256]).flatten().astype(np.float32)
    s = float(h.sum())
    return h / s if s > 0 else None


def appearance_signature(hsv, lms):
    """上半身(肩〜腰)と太もも(腰〜膝)の色ヒストグラム。取れない部位は含めない。"""
    H, W = hsv.shape[:2]
    sig = {}
    if all(lms[i, 3] >= 0.3 for i in (11, 12, 23, 24)):
        sh_y = float(lms[[11, 12], 1].mean()) * H
        hp_y = float(lms[[23, 24], 1].mean()) * H
        th = hp_y - sh_y
        if th > 6:
            xs = lms[[11, 12, 23, 24], 0] * W
            pad = 0.15 * th
            sig["torso"] = _region_hist(hsv, xs.min() - pad, sh_y + 0.1 * th, xs.max() + pad, hp_y - 0.05 * th)
    if all(lms[i, 3] >= 0.3 for i in (23, 24, 25, 26)):
        hp_y = float(lms[[23, 24], 1].mean()) * H
        kn_y = float(lms[[25, 26], 1].mean()) * H
        tl = kn_y - hp_y
        if tl > 6:
            xs = lms[[23, 24, 25, 26], 0] * W
            sig["thigh"] = _region_hist(hsv, xs.min() - 0.1 * tl, hp_y + 0.1 * tl, xs.max() + 0.1 * tl, kn_y - 0.1 * tl)
    sig = {k: v for k, v in sig.items() if v is not None}
    return sig or None


def appearance_distance(a, b):
    """0(同じ) 〜 1(全く違う)。比較できる部位が無ければ None。"""
    import cv2

    if not a or not b:
        return None
    ds = [float(cv2.compareHist(a[k], b[k], cv2.HISTCMP_BHATTACHARYYA)) for k in ("torso", "thigh")
          if k in a and k in b]
    return float(np.mean(ds)) if ds else None


def _blend_sig(old, new, alpha=0.15):
    if not old:
        return new
    if not new:
        return old
    out = dict(old)
    for k, v in new.items():
        if k in out:
            m = (1 - alpha) * out[k] + alpha * v
            out[k] = m / float(m.sum())
        else:
            out[k] = v
    return out


# ----------------------------------------------------------------------
# 描画
# ----------------------------------------------------------------------
def _draw_ground(img, ground_y, tol, alpha=0.2):
    import cv2

    h, w = img.shape[:2]
    y_line = min(h - 1, int(ground_y * h))
    y_top = min(h - 1, max(0, int((ground_y - tol) * h)))
    roi = img[y_top:, :]
    green = np.zeros_like(roi)
    green[:] = (0, 200, 0)
    cv2.addWeighted(green, alpha, roi, 1 - alpha, 0, roi)
    cv2.line(img, (0, y_line), (w, y_line), (0, 255, 0), 2)
    cv2.line(img, (0, y_top), (w, y_top), (0, 200, 0), 1)
    cv2.putText(img, "GROUND ZONE", (6, min(h - 6, y_top + 16)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)


def _draw_person(img, lms, color, thick):
    import cv2

    h, w = img.shape[:2]
    for a, b in POSE_CONNECTIONS:
        if lms[a, 3] > 0.3 and lms[b, 3] > 0.3:
            cv2.line(img, (int(lms[a, 0] * w), int(lms[a, 1] * h)),
                     (int(lms[b, 0] * w), int(lms[b, 1] * h)), color, thick, cv2.LINE_AA)
    for i in range(11, 33):
        if lms[i, 3] > 0.3:
            cv2.circle(img, (int(lms[i, 0] * w), int(lms[i, 1] * h)), thick + 1, color, -1, cv2.LINE_AA)


def _draw_ankles(img, lms, ground_y, tol):
    import cv2

    h, w = img.shape[:2]
    for i in (27, 28):
        if lms[i, 3] < 0.2:
            continue
        c = (int(lms[i, 0] * w), int(lms[i, 1] * h))
        in_zone = lms[i, 1] > ground_y - tol
        if in_zone:
            cv2.circle(img, c, 9, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(img, c, 9, (0, 255, 255), 2, cv2.LINE_AA)


def _label(img, text, org, color):
    import cv2

    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


TARGET_COLOR = (0, 140, 255)  # BGR: オレンジ
OTHER_COLOR = (170, 170, 170)


def render_preview(frame_bgr, people, target_idx, ground_y, tol, frame_idx, in_range, banner="", roles=True):
    """プレビュー画像(RGB)を返す。ターゲット/他の人/足首/接地ゾーン/範囲内外を描画。"""
    import cv2

    img = frame_bgr.copy()
    h, w = img.shape[:2]
    _draw_ground(img, ground_y, tol)
    for i, p in enumerate(people):
        is_t = (i == target_idx)
        _draw_person(img, p, TARGET_COLOR if is_t else OTHER_COLOR, 3 if is_t else 1)
        head = (int(p[0, 0] * w) if p[0, 3] > 0.3 else int(person_center(p)[0] * w),
                max(14, int((p[0, 1] if p[0, 3] > 0.3 else person_center(p)[1] - 0.1) * h) - 18))
        if is_t:
            txt = f"#{i + 1} TARGET"
        elif roles:
            txt = f"#{i + 1} IGNORE"
        else:
            txt = f"#{i + 1}"
        _label(img, txt, head, TARGET_COLOR if is_t else (220, 220, 220))
        if is_t:
            _draw_ankles(img, p, ground_y, tol)
    border = (0, 200, 0) if (in_range or in_range is None) else (0, 0, 255)
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), border, 6)
    if in_range is None:  # 開始フレームだけを見せるプレビュー
        state_txt = "START FRAME"
    else:
        state_txt = "IN RANGE" if in_range else "OUT OF RANGE"
    _label(img, f"Frame {frame_idx}  {state_txt}  {banner}".rstrip(), (10, 24), border)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _make_thumb(img, lms, size=110):
    """ロック中の人の切り抜き(確認用)。JPEGバイト列を返す。"""
    import cv2

    H, W = img.shape[:2]
    ok = lms[:, 3] > 0.3
    if ok.sum() < 4:
        return None
    x0, x1 = lms[ok, 0].min() * W, lms[ok, 0].max() * W
    y0, y1 = lms[ok, 1].min() * H, lms[ok, 1].max() * H
    px, py = 0.25 * (x1 - x0) + 10, 0.12 * (y1 - y0) + 6
    x0, x1 = max(0, int(x0 - px)), min(W, int(x1 + px))
    y0, y1 = max(0, int(y0 - py)), min(H, int(y1 + py))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    crop = img[y0:y1, x0:x1]
    scale = size / float(crop.shape[0])
    crop = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), size), interpolation=cv2.INTER_AREA)
    ok_j, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return buf.tobytes() if ok_j else None


# ----------------------------------------------------------------------
# ターゲット追跡 (最初の走者1人だけをロックオン)
# ----------------------------------------------------------------------
class TargetTracker:
    """最初の走者「だけ」をロックオンし、それ以外の人物は一切追わない。

    - acquire(): ロック前。プレビューで指定した人(target_point)に近い人だけにロックする。
      指定が無ければ最初に映った人。別の人への自動切り替えはしない。
    - roi(): 前フレームの骨格の周辺(切り出し範囲)を返す。ここだけを検出する。
    - match(): 候補の中から「位置・体の大きさ・服の色」がロック中の人と合う1人だけを選ぶ。
    - hit()/miss(): 状態の更新。見失いが続く/画面端で消えたら ended=True。二度と再取得しない。
    """

    def __init__(self, target_point=None, fps=30.0, strictness="標準"):
        cfg = STRICTNESS.get(strictness, STRICTNESS["標準"])
        self.target_point = target_point
        self.fps = fps
        self.app_thr = cfg["app_thr"]
        self.lost_limit = max(4, int(fps * cfg["lost_sec"]))
        self.edge_lost_limit = max(2, int(fps * 0.1))
        self.max_wait = max(10, int(fps * 1.0))
        self.locked = False
        self.ended = False
        self.end_reason = ""
        self.center = None
        self.height = None
        self.vel = (0.0, 0.0)
        self.lost = 0
        self.wait = 0
        self.sig = None    # 追従して少しずつ更新する見た目
        self.sig0 = None   # ロック時の見た目 (基準。更新しない)
        self.lms = None
        self._blocked = False  # 直近の見失いが「近くの別の人に隠れた/別人しかいない」状態か

    def _end(self, reason):
        self.ended = True
        self.end_reason = reason

    # ---------- ロック前 ----------
    def _lock(self, p, hsv):
        self.locked = True
        self.center = person_center(p)
        h = person_height(p)
        self.height = h if h > 0 else None
        self.vel = (0.0, 0.0)
        self.lost = 0
        self.lms = p
        self.sig = appearance_signature(hsv, p) if hsv is not None else None
        self.sig0 = self.sig

    def acquire(self, people, hsv):
        """ロックできたらその人の index、まだなら None。"""
        self.wait += 1
        cands = [i for i, p in enumerate(people) if person_quality(p) >= 0.4]
        if self.target_point is not None:
            if cands:
                ds = {i: math.hypot(person_center(people[i])[0] - self.target_point[0],
                                    person_center(people[i])[1] - self.target_point[1]) for i in cands}
                i = min(ds, key=ds.get)
                if ds[i] < 0.25:
                    self._lock(people[i], hsv)
                    return i
            if self.wait > self.max_wait:
                self._end("指定したターゲットを開始位置で検出できなかったため解析を終了しました")
            return None
        if cands:  # 指定なし: 最初に映った人（同時に複数なら大きい人）
            big = [i for i in cands if person_height(people[i]) >= 0.12] or cands
            i = max(big, key=lambda j: person_height(people[j]))
            self._lock(people[i], hsv)
            return i
        return None

    # ---------- ロック後 ----------
    def roi(self, H, W):
        """前フレームの骨格の周辺の切り出し範囲 (x0, y0, x1, y1) ピクセル。"""
        p = self.lms
        ok = p[:, 3] > 0.3
        if ok.sum() < 4:
            return (0, 0, W, H)
        k = min(self.lost + 1, 4)
        xs, ys = p[ok, 0], p[ok, 1]
        cx = (xs.min() + xs.max()) / 2 + self.vel[0] * k
        cy = (ys.min() + ys.max()) / 2 + self.vel[1] * k
        bw, bh = xs.max() - xs.min(), ys.max() - ys.min()
        grow = 1.0 + 0.25 * min(self.lost, 4)
        ph = max(bh * H * 1.5, 0.25 * H) * grow
        pw = max(bw * W * 1.6, ph * 0.9) * grow
        x0 = int(max(0, cx * W - pw / 2))
        x1 = int(min(W, cx * W + pw / 2))
        y0 = int(max(0, cy * H - ph / 2))
        y1 = int(min(H, cy * H + ph / 2))
        if x1 - x0 < 64 or y1 - y0 < 64:
            return (0, 0, W, H)
        return (x0, y0, x1, y1)

    def match(self, people, hsv):
        """ロック中の人と合う候補の index。合う人がいなければ None（状態は変更しない）。"""
        if not people or not self.locked or self.ended:
            return None
        k = min(self.lost + 1, 4)
        pred = (self.center[0] + self.vel[0] * k, self.center[1] + self.vel[1] * k)
        gate = min(0.12 + 0.02 * self.lost, 0.22)
        best, best_cost = None, 1e9
        self._blocked = False
        for i, p in enumerate(people):
            if person_quality(p) < 0.3:
                continue
            c = person_center(p)
            d = math.hypot(c[0] - pred[0], c[1] - pred[1])
            if d > gate:
                continue
            h = person_height(p)
            if self.height and h > 0 and not (0.65 <= h / self.height <= 1.55):
                self._blocked = True
                continue  # 体の大きさが違う = 別人
            sig = appearance_signature(hsv, p)
            ad = appearance_distance(self.sig, sig)
            ad0 = appearance_distance(self.sig0, sig)  # ロック時の基準とも照合 (見た目の混入によるドリフト防止)
            if (ad is not None and ad > self.app_thr) or (ad0 is not None and ad0 > min(1.0, self.app_thr + 0.15)):
                self._blocked = True
                continue  # 服の色が違う = 別人
            cost = d / gate + (1.5 * ad if ad is not None else 0.5)
            if cost < best_cost:
                best, best_cost = i, cost
        return best

    def hit(self, p, hsv, crowded=False):
        """採用した人で状態を更新。crowded=True (近くに別の人がいる) の間は見た目を更新しない。"""
        new = person_center(p)
        kk = self.lost + 1
        self.vel = (0.5 * self.vel[0] + 0.5 * (new[0] - self.center[0]) / kk,
                    0.5 * self.vel[1] + 0.5 * (new[1] - self.center[1]) / kk)
        self.center = new
        h = person_height(p)
        if h > 0:
            self.height = h if not self.height else 0.8 * self.height + 0.2 * h
        sig = appearance_signature(hsv, p)
        if self.sig0 is None and sig:
            self.sig0 = sig
        ad = appearance_distance(self.sig, sig)
        if sig and not crowded and (ad is None or ad < 0.35):  # 単独で、明らかに同じ人のときだけゆっくり更新
            self.sig = _blend_sig(self.sig, sig, alpha=0.08)
        self.lms = p
        self.lost = 0
        self._blocked = False

    def miss(self):
        self.lost += 1
        cx = self.center[0]
        exiting = (cx > 0.85 and self.vel[0] > 0.003) or (cx < 0.15 and self.vel[0] < -0.003)
        # 近くの別の人に隠れている間は少しだけ長く待つ (追い抜き・重なり対策)。ただし別人には乗り移らない
        limit = self.lost_limit * 2 if self._blocked else self.lost_limit
        if exiting and self.lost >= self.edge_lost_limit:
            self._end("ターゲットがフレームアウトしたため解析を終了しました")
        elif self.lost > limit:
            self._end("ターゲットを見失ったため（フレームアウトとみなし）解析を終了しました")


# ----------------------------------------------------------------------
# メイン解析
# ----------------------------------------------------------------------
def _reencode_h264(path):
    """ffmpeg があればブラウザ再生可能なH.264へ変換 (失敗したら元ファイルのまま)。"""
    if not shutil.which("ffmpeg"):
        return path, False
    out = path.replace(".mp4", "_h264.mp4")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", path, "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", out],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300)
        os.remove(path)
        return out, True
    except Exception:
        if os.path.exists(out):
            os.remove(out)
        return path, False


def analyze_sprint(input_video_path, ground_y_normalized=0.8, start_frame=0, end_frame=None,
                   target_point=None, ground_tol=0.05, max_width=640, dry_run=False,
                   progress_callback=None, max_captures=40, strictness="標準", max_steps=3, model="full"):
    """
    戻り値: (output_video_path or None, report_text, extras)
      extras = {captures, thumbs, trajectory, track, lock_start, lock_end, end_reason, codec, playable, size, contacts}
    dry_run=True の場合は動画・静止画を出力せず、追跡と接地検出のみ行う。
    """
    import cv2

    info = get_video_info(input_video_path)
    fps = info["fps"]
    total = info["frames"]
    start_frame = max(0, int(start_frame))
    end_frame = total - 1 if end_frame is None else min(int(end_frame), total - 1)
    if end_frame < start_frame:
        raise ValueError("解析範囲の終了フレームが開始フレームより前になっています。")

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError("動画を開けませんでした。")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    landmarker = _make_landmarker(video_mode=True, num_poses=4, kind=model)  # ロック前・再取得用(画面全体)
    tracker = TargetTracker(target_point, fps, strictness)

    writer = None
    out_path = None
    codec_used = None
    size = None
    captures = []
    thumbs = []
    trajectory = []
    track = {}
    summary_report = []
    skipped = []

    grounded = False
    air_frames = 0
    lock_start = None
    lock_end = None
    end_reason = "指定した終了フレームまで解析しました"
    last_ts = -1
    n_done = 0
    n_matched = 0
    thumb_step = max(1, int(fps * 0.2))
    n_all = end_frame - start_frame + 1

    try:
        idx = start_frame
        while idx <= end_frame:
            ok, raw = cap.read()
            if not ok:
                end_reason = "動画の終端に到達しました"
                break
            img = resize_frame(raw, max_width)  # 表示・描画・全体検出用 (縮小)。切り出し検出は元解像度の raw から行う
            H, W = img.shape[:2]

            ts = max(int(n_done * 1000 / fps), last_ts + 1)
            last_ts = ts

            lms = None  # このフレームのターゲットの骨格 (見つからなければ None)
            if not tracker.locked:
                people = _detect(landmarker, img, ts)
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) if people else None
                ti = tracker.acquire(people, hsv)
                if ti is not None:
                    lms = people[ti]
            else:
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                # ① 前フレームの骨格の周辺だけを検出 (ROI追跡)
                x0, y0, x1, y1 = tracker.roi(H, W)
                sx, sy = raw.shape[1] / float(W), raw.shape[0] / float(H)
                people = detect_people_roi(
                    raw, (int(x0 * sx), int(y0 * sy), int(x1 * sx), int(y1 * sy)), kind=model)
                ti = tracker.match(people, hsv)
                # ② 取れなければ画面全体で探し直す (同じ人かどうかは位置・大きさ・服の色で厳密に確認)
                if ti is None:
                    full = _detect(landmarker, img, ts)
                    j = tracker.match(full, hsv)
                    if j is not None:
                        people, ti = full, j
                if ti is not None:
                    lms = people[ti]
                    cx0, cy0 = person_center(lms)
                    crowded = any(math.hypot(person_center(q)[0] - cx0, person_center(q)[1] - cy0) < 0.10
                                  for j, q in enumerate(people) if j != ti and person_quality(q) >= 0.3)
                    tracker.hit(lms, hsv, crowded)
                else:
                    tracker.miss()

            if tracker.ended:
                end_reason = tracker.end_reason
                if lock_end is not None:
                    end_reason += f"（最後に検出: フレーム {lock_end}）"
                break

            touchdown_step = None
            ankle_y_val = None
            if lms is not None:
                if lock_start is None:
                    lock_start = idx
                lock_end = idx
                track[idx] = person_center(lms)
                ankle_y_val = lowest_ankle_y(lms)

                if n_matched % thumb_step == 0 and len(thumbs) < 30:
                    tb = _make_thumb(img, lms)  # 描画前の画像から切り抜く
                    if tb is not None:
                        thumbs.append((idx, tb))
                n_matched += 1

                # 足首が低い(=接地側)の脚を採用
                if lms[27, 1] >= lms[28, 1]:
                    side, sh, hp, kn, an = "Left", lms[11], lms[23], lms[25], lms[27]
                else:
                    side, sh, hp, kn, an = "Right", lms[12], lms[24], lms[26], lms[28]

                if an[3] >= 0.2:
                    if an[1] > ground_y_normalized - ground_tol:
                        air_frames = 0
                        if not grounded:
                            grounded = True
                            knee_angle = calculate_angle(hp, kn, an)
                            hip_angle = calculate_angle(sh, hp, kn)
                            vis_ok = min(sh[3], hp[3], kn[3], an[3]) >= 0.3
                            # 緩いフィルター: 明らかな誤検出(極端な角度・低信頼)のみ除外
                            plausible = vis_ok and 90 <= knee_angle <= 180 and 110 <= hip_angle <= 180
                            if plausible:
                                k_eval, k_res, k_drill = evaluate_knee(knee_angle)
                                h_eval, h_res, h_drill = evaluate_hip(hip_angle)
                                summary_report.append({
                                    "step": len(summary_report) + 1, "frame": idx, "leg": side,
                                    "knee_ang": knee_angle, "knee_eval": k_eval,
                                    "knee_res": k_res, "knee_drill": k_drill,
                                    "hip_ang": hip_angle, "hip_eval": h_eval,
                                    "hip_res": h_res, "hip_drill": h_drill,
                                })
                                touchdown_step = summary_report[-1]
                            else:
                                skipped.append(idx)
                    else:
                        air_frames += 1
                        if air_frames >= 2:
                            grounded = False

            trajectory.append((idx, ankle_y_val))

            if not dry_run:
                disp = img  # 縮小済みフレームをそのまま加工
                _draw_ground(disp, ground_y_normalized, ground_tol)
                if lms is not None:  # ターゲット以外は描画もしない
                    _draw_person(disp, lms, TARGET_COLOR, 2)
                    _draw_ankles(disp, lms, ground_y_normalized, ground_tol)
                    _label(disp, "TARGET", (10, 22), TARGET_COLOR)
                if writer is None:
                    size = (disp.shape[1], disp.shape[0])
                    out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
                    for cc in ("avc1", "mp4v"):
                        w_ = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*cc), fps, size)
                        if w_.isOpened():
                            writer, codec_used = w_, cc
                            break
                        w_.release()
                    if writer is None:
                        raise RuntimeError("動画書き出しを初期化できませんでした。")
                writer.write(disp)

                if touchdown_step is not None and len(captures) < max_captures:
                    cap_img = disp.copy()
                    _label(cap_img, f"STEP {touchdown_step['step']} ({touchdown_step['leg']})  "
                                    f"knee {touchdown_step['knee_ang']}  trunk {touchdown_step['hip_ang']}",
                           (10, cap_img.shape[0] - 12), (0, 255, 255))
                    ok_j, buf = cv2.imencode(".jpg", cap_img, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ok_j:
                        captures.append((touchdown_step["step"], buf.tobytes(),
                                         f"第{touchdown_step['step']}歩目 / Frame {idx} / {touchdown_step['leg']}脚"))

            # 最初のN歩だけ解析して終了 (ロックした人の接地を max_steps 歩分取得したら打ち切り)
            if max_steps and len(summary_report) >= max_steps:
                end_reason = f"ロックした走者の最初の{max_steps}歩分の接地を取得したため解析を終了しました"
                break

            n_done += 1
            idx += 1
            if n_done % 60 == 0:
                gc.collect()
            if progress_callback and n_done % 5 == 0:
                progress_callback(min(1.0, n_done / n_all), f"フレーム {idx}/{end_frame + 1} を処理中...")
    finally:
        cap.release()
        landmarker.close()
        if writer is not None:
            writer.release()
        gc.collect()

    playable = False
    if out_path is not None:
        if codec_used == "avc1":
            playable = True
        else:
            out_path, playable = _reencode_h264(out_path)

    # ------------------------------ レポート ------------------------------
    report_text = "### 🏃‍♂️ スプリント バイオメカニクス 診断レポート（エリート基準連動型）\n\n"
    report_text += "> 📌 **【ラルフ・マン エリート基準（比較用）】**\n"
    report_text += "> * **膝角度（接地時）:** `152〜155度` が理想\n"
    report_text += "> * **体幹角度（接地時）:** `170〜176度` が理想\n\n"
    if lock_start is not None:
        report_text += (f"> 🎯 **解析対象:** 最初にロックオンした走者 1名のみ（フレーム {lock_start}〜{lock_end}）\n"
                        f"> * {end_reason}\n"
                        f"> * それ以外の人物は追跡・解析の対象外です\n\n")
    else:
        report_text += f"> ⚠️ ターゲットをロックオンできませんでした。{end_reason}\n\n"
    report_text += "---\n\n"

    if len(summary_report) == 0:
        report_text += "⚠️ **エリート基準の評価対象となる安定した接地データが検出されませんでした。**\n"
        report_text += "\n接地ラインの位置・接地ゾーンの幅・解析範囲を調整して再実行してください。\n"
    else:
        for data in summary_report:
            report_text += f"**【第 {data['step']} 歩目】（{data['leg']} 脚 / Frame {data['frame']}）**\n"
            report_text += (f"* **膝角度:** `計測値 {data['knee_ang']}度` （基準: 152〜155度） "
                            f"-> **{data['knee_eval']}**\n")
            if data["knee_res"]:
                report_text += f"  * *原因:* {data['knee_res']}\n  * *対策:* {data['knee_drill']}\n"
            report_text += (f"* **体幹角度:** `計測値 {data['hip_ang']}度` （基準: 170〜176度） "
                            f"-> **{data['hip_eval']}**\n")
            if data["hip_res"]:
                report_text += f"  * *原因:* {data['hip_res']}\n  * *対策:* {data['hip_drill']}\n"
            report_text += "\n---\n"
    if skipped:
        report_text += (f"\n※ 低信頼・極端な角度のため除外した接地候補: {len(skipped)}件 "
                        f"（Frame {', '.join(map(str, skipped[:10]))}{' ...' if len(skipped) > 10 else ''}）\n")

    extras = {
        "captures": captures,
        "thumbs": thumbs,
        "trajectory": trajectory,
        "track": track,
        "lock_start": lock_start,
        "lock_end": lock_end,
        "end_reason": end_reason,
        "codec": codec_used,
        "playable": playable,
        "size": size,
        "contacts": [d["frame"] for d in summary_report],
    }
    return out_path, report_text, extras
