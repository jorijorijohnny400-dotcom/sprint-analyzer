"""スプリント バイオメカニクス解析エンジン (Streamlit Community Cloud 対応版)

- MediaPipe Tasks API (PoseLandmarker) を使用 (mp.solutions は使わない)
- cv2 / mediapipe は関数内で遅延インポート (起動時のフリーズ防止)
- フレームは横幅 max_width(既定640px) に縮小して処理し、メモリを節約
- 最初に映った (またはプレビューで指定した) 走者をロックオンし、その人の接地だけを検出
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

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
MODEL_PATH = os.path.join(tempfile.gettempdir(), "pose_landmarker_lite.task")

POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
    (27, 29), (29, 31), (27, 31), (28, 30), (30, 32), (28, 32),
]

_IMG_LM = None
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


def ensure_model():
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1_000_000:
        return MODEL_PATH
    tmp = MODEL_PATH + ".part"
    urllib.request.urlretrieve(MODEL_URL, tmp)
    os.replace(tmp, MODEL_PATH)
    return MODEL_PATH


# ----------------------------------------------------------------------
# PoseLandmarker (Tasks API)
# ----------------------------------------------------------------------
def _make_landmarker(video_mode, num_poses):
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    mode = vision.RunningMode.VIDEO if video_mode else vision.RunningMode.IMAGE
    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=ensure_model()),
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


def detect_people_image(frame_bgr):
    """プレビュー用の単一画像検出。左→右の順に並べて返す。"""
    global _IMG_LM
    with _IMG_LOCK:
        if _IMG_LM is None:
            _IMG_LM = _make_landmarker(video_mode=False, num_poses=5)
        people = _detect(_IMG_LM, frame_bgr)
    people = [p for p in people if person_quality(p) >= 0.3]
    people.sort(key=lambda p: person_center(p)[0])
    return people


def match_center(people, center, max_d=0.2):
    """記録済みのターゲット中心に最も近い人の index（遠ければ None）。"""
    if not people or center is None:
        return None
    d = [math.hypot(person_center(p)[0] - center[0], person_center(p)[1] - center[1]) for p in people]
    i = int(np.argmin(d))
    return i if d[i] <= max_d else None


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


def find_first_person_frame(path, start=0, step=3, max_seconds=20, max_width=480):
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
        for p in detect_people_image(img):
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
    border = (0, 200, 0) if in_range else (0, 0, 255)
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), border, 6)
    _label(img, f"Frame {frame_idx}  {'IN RANGE' if in_range else 'OUT OF RANGE'}  {banner}", (10, 24), border)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ----------------------------------------------------------------------
# ターゲット追跡 (最初の走者をロックオン)
# ----------------------------------------------------------------------
class TargetTracker:
    """最初の走者「だけ」をロックオンし、それ以外の人物は一切追わない。

    - ロック前: プレビューで指定した人(target_point)に近い人だけにロックする。
      指定が無ければ、最初に映った人にロックする。別の人へ自動で切り替えることはしない。
    - ロック後: 予測位置の近くで、体の大きさも近い人だけを同一人物とみなす。
      見失う時間が長くなっても照合範囲は広げない。
    - 終了: フレーム端で消えた、または一定時間見失ったら ended=True。
      一度終了したら二度と再取得しない（後から映る別の人に乗り移らない）。
    """

    def __init__(self, target_point=None, fps=30.0):
        self.target_point = target_point
        self.fps = fps
        self.locked = False
        self.ended = False
        self.end_reason = ""
        self.center = None
        self.height = None
        self.vel = (0.0, 0.0)
        self.lost = 0
        self.wait = 0
        self.lost_limit = max(4, int(fps * 0.3))
        self.edge_lost_limit = max(2, int(fps * 0.1))
        self.max_wait = max(10, int(fps * 1.0))

    def _lock(self, people, i):
        self.locked = True
        self.center = person_center(people[i])
        h = person_height(people[i])
        self.height = h if h > 0 else None
        self.vel = (0.0, 0.0)
        self.lost = 0
        return i

    def _end(self, reason):
        self.ended = True
        self.end_reason = reason

    def update(self, people):
        """ターゲットの index を返す。見つからなければ None。"""
        if self.ended:
            return None

        # ---------- ロック前 ----------
        if not self.locked:
            self.wait += 1
            cands = [i for i, p in enumerate(people) if person_quality(p) >= 0.4]
            if self.target_point is not None:
                if cands:
                    ds = {i: math.hypot(person_center(people[i])[0] - self.target_point[0],
                                        person_center(people[i])[1] - self.target_point[1])
                          for i in cands}
                    i = min(ds, key=ds.get)
                    if ds[i] < 0.25:
                        return self._lock(people, i)
                if self.wait > self.max_wait:
                    self._end("指定したターゲットを開始位置で検出できなかったため解析を終了しました")
                return None
            # 指定なし: 最初に映った人（同時に複数なら大きい人）
            if cands:
                big = [i for i in cands if person_height(people[i]) >= 0.12] or cands
                return self._lock(people, max(big, key=lambda j: person_height(people[j])))
            return None

        # ---------- ロック後 ----------
        k = min(self.lost + 1, 4)
        pred = (self.center[0] + self.vel[0] * k, self.center[1] + self.vel[1] * k)
        gate = min(0.12 + 0.02 * self.lost, 0.22)

        best, best_d = None, 1e9
        for i, p in enumerate(people):
            if person_quality(p) < 0.3:
                continue
            c = person_center(p)
            d = math.hypot(c[0] - pred[0], c[1] - pred[1])
            h = person_height(p)
            if self.height and h > 0 and not (0.55 <= h / self.height <= 1.8):
                continue  # 体の大きさが違う = 別人
            if d < best_d:
                best, best_d = i, d

        if best is not None and best_d <= gate:
            new = person_center(people[best])
            kk = self.lost + 1
            self.vel = (0.5 * self.vel[0] + 0.5 * (new[0] - self.center[0]) / kk,
                        0.5 * self.vel[1] + 0.5 * (new[1] - self.center[1]) / kk)
            self.center = new
            h = person_height(people[best])
            if h > 0:
                self.height = h if not self.height else 0.8 * self.height + 0.2 * h
            self.lost = 0
            return best

        # 見失った
        self.lost += 1
        cx = self.center[0]
        exiting = (cx > 0.85 and self.vel[0] > 0.003) or (cx < 0.15 and self.vel[0] < -0.003)
        if exiting and self.lost >= self.edge_lost_limit:
            self._end("ターゲットがフレームアウトしたため解析を終了しました")
        elif self.lost > self.lost_limit:
            self._end("ターゲットを見失ったため（フレームアウトとみなし）解析を終了しました")
        return None


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
                   progress_callback=None, max_captures=40):
    """
    戻り値: (output_video_path or None, report_text, extras)
      extras = {captures, trajectory, lock_start, lock_end, end_reason, codec, playable, size}
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

    landmarker = _make_landmarker(video_mode=True, num_poses=4)
    tracker = TargetTracker(target_point, fps)

    writer = None
    out_path = None
    codec_used = None
    size = None
    captures = []
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
    n_all = end_frame - start_frame + 1

    try:
        idx = start_frame
        while idx <= end_frame:
            ok, img = cap.read()
            if not ok:
                end_reason = "動画の終端に到達しました"
                break
            img = resize_frame(img, max_width)

            ts = max(int(n_done * 1000 / fps), last_ts + 1)
            last_ts = ts
            people = _detect(landmarker, img, ts)
            ti = tracker.update(people)

            if tracker.ended:
                end_reason = tracker.end_reason
                if lock_end is not None:
                    end_reason += f"（最後に検出: フレーム {lock_end}）"
                break

            touchdown_step = None
            ankle_y_val = None
            if ti is not None:
                lms = people[ti]
                if lock_start is None:
                    lock_start = idx
                lock_end = idx
                track[idx] = person_center(lms)
                ankle_y_val = lowest_ankle_y(lms)

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
                if ti is not None:  # ターゲット以外は描画もしない
                    _draw_person(disp, people[ti], TARGET_COLOR, 2)
                    _draw_ankles(disp, people[ti], ground_y_normalized, ground_tol)
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
