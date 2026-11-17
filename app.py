import gc
import os
import tempfile

import streamlit as st

# engine は cv2 / mediapipe を関数内で遅延インポートするため、起動時にフリーズしない
import engine

st.set_page_config(
    page_title="スプリントバイオメカニクス解析システム",
    page_icon="🏃‍♂️",
    layout="wide",
)

# ----------------------------------------------------------------------
# 固定設定（画面からは変更しない）
# ----------------------------------------------------------------------
PREVIEW_WIDTH = 640      # 解析時の横幅（メモリ対策で自動縮小）
STRICTNESS = "緩い"       # ターゲット追跡の厳しさ（常に「緩い」）
MAX_STEPS = 3            # ロックした走者の最初の何歩分の接地を解析するか
MODEL_KIND = "full"      # 姿勢推定モデル（取得できなければ自動で lite にフォールバック）

# Streamlit の版による引数の違いを吸収（新しい版は width="stretch"、古い版は use_container_width=True）
try:
    _ver = tuple(int(x) for x in st.__version__.split(".")[:2])
except Exception:
    _ver = (0, 0)
STRETCH = {"width": "stretch"} if _ver >= (1, 50) else {"use_container_width": True}

st.title("🏃‍♂️ スプリント バイオメカニクス 診断システム")
st.markdown("横から撮影したスプリント動画をアップロードし、ラルフ・マンのエリート基準に基づいた動作解析を行います。")


# ----------------------------------------------------------------------
# キャッシュ付きフレーム取得 + 人物検出 (プレビュー用)
# ----------------------------------------------------------------------
@st.cache_data(show_spinner=False, max_entries=24)
def cached_frame_people(path, idx, kind=MODEL_KIND):
    frame = engine.read_frame(path, idx, PREVIEW_WIDTH)
    if frame is None:
        return None, []
    return frame, engine.detect_people_image(frame, kind)


def discard_video():
    p = st.session_state.get("video_path")
    if p and os.path.exists(p):
        os.remove(p)
    for k in ("video_path", "video_id", "info", "result", "start_f", "ground_y", "tol", "target_no", "_ankle_y"):
        st.session_state.pop(k, None)
    cached_frame_people.clear()
    gc.collect()


def discard_and_reset_uploader():
    """破棄ボタン用: データを消し、アップロード欄も空に戻す（残ったファイルが再保存されないように）。"""
    discard_video()
    st.session_state["uploader_n"] = st.session_state.get("uploader_n", 0) + 1


# ----------------------------------------------------------------------
# ステップ1: アップロード
# ----------------------------------------------------------------------
st.header("① 動画を選ぶ")
uploaded = st.file_uploader("スプリント動画 (.mp4, .mov)", type=["mp4", "mov"], label_visibility="collapsed",
                            key=f"uploader_{st.session_state.get('uploader_n', 0)}")

if uploaded is None:
    if "video_path" in st.session_state:
        discard_video()
    st.info("横から撮影した走りの動画をアップロードしてください。")
    st.stop()

vid_id = f"{uploaded.name}-{uploaded.size}"
if st.session_state.get("video_id") != vid_id:
    discard_video()
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tf.write(uploaded.getbuffer())
    tf.close()
    st.session_state["video_path"] = tf.name
    st.session_state["video_id"] = vid_id
    try:
        st.session_state["info"] = engine.get_video_info(tf.name)
    except Exception as e:
        st.error(str(e))
        discard_video()
        st.stop()

path = st.session_state["video_path"]
info = st.session_state["info"]
n_frames = info["frames"]
fps = info["fps"]
last = n_frames - 1

st.caption(f"{info['width']}×{info['height']} / {fps:.1f}fps / {n_frames}フレーム "
           f"（解析時は横幅{PREVIEW_WIDTH}pxに自動縮小します）")

st.session_state.setdefault("start_f", 0)
st.session_state.setdefault("ground_y", 0.80)
st.session_state.setdefault("tol", 0.05)


# ----------------------------------------------------------------------
# コールバック
# ----------------------------------------------------------------------
def cb_find_first():
    with st.spinner("最初に人が映るフレームを探しています..."):
        f = engine.find_first_person_frame(path, 0, kind=MODEL_KIND)
    if f is None:
        st.session_state["_msg"] = ("warning", "人物を検出できませんでした。スライダーで手動で合わせてください。")
    else:
        st.session_state["start_f"] = int(f)
        st.session_state["_msg"] = ("success", f"フレーム {f} で最初の人物を検出し、解析の開始位置に設定しました。")


def cb_fit_ground():
    y = st.session_state.get("_ankle_y")
    if y is not None:
        st.session_state["ground_y"] = float(min(0.99, max(0.30, round(y, 3))))


# ----------------------------------------------------------------------
# ステップ2・3: 走者と接地ラインを合わせる（プレビュー = 解析の開始フレーム）
# ----------------------------------------------------------------------
st.header("② 走者と地面の高さを合わせる")
st.caption("左のプレビューは「解析を始める位置」の画面です。この画面の内容どおりに解析されます。")

if st.session_state.get("_msg"):
    kind_msg, text_msg = st.session_state.pop("_msg")
    (st.success if kind_msg == "success" else st.warning)(text_msg)

left, right = st.columns([3, 2], gap="large")

# --- 右: 操作パネル ---
with right:
    with st.container(border=True):
        st.markdown("##### 🅰 解析を始める位置")
        st.slider("開始フレーム（走者が映った位置に合わせる）", 0, last, key="start_f")
        st.button("🔍 最初に人が映る位置を自動で探す", on_click=cb_find_first, **STRETCH)
        start_f = int(st.session_state["start_f"])
        st.caption(f"{start_f / fps:.2f}秒（フレーム {start_f}）から解析します。"
                   "走者がフレームアウトするか、最初の3歩の接地を取れた時点で自動的に終了します。")

# 開始フレームの人物検出
frame, people = cached_frame_people(path, start_f)

with right:
    with st.container(border=True):
        st.markdown("##### 🅱 解析する走者")
        target_point = None
        t_idx = None
        if frame is None:
            st.error("このフレームを読み込めませんでした。")
        elif not people:
            st.warning("この位置では人物が見つかりません。開始フレームを動かすか、上の自動検出ボタンを押してください。")
        else:
            if len(people) > 1:
                labels = [f"#{i + 1}（左から{i + 1}人目）" for i in range(len(people))]
                default_i = engine.default_target_index(people)
                if st.session_state.get("target_no", None) is None or st.session_state["target_no"] >= len(people):
                    st.session_state["target_no"] = default_i
                sel = st.radio("追跡する人を選んでください（選んだ1人だけを追います）", labels,
                               index=st.session_state["target_no"])
                t_idx = labels.index(sel)
                st.session_state["target_no"] = t_idx
            else:
                t_idx = 0
                st.success("この走者 1人を追跡します。")
            target_point = engine.person_center(people[t_idx])

    with st.container(border=True):
        st.markdown("##### 🅲 地面の高さ（緑のライン）")
        st.slider("緑のラインを、走者の足元の高さに合わせる", 0.30, 0.99, step=0.005, key="ground_y")
        st.button("👟 足首の高さに自動で合わせる", on_click=cb_fit_ground, **STRETCH,
                  disabled=(t_idx is None))
        st.slider("接地とみなす幅（広いほど接地を拾いやすい）", 0.02, 0.15, step=0.005, key="tol")

ground_y = st.session_state["ground_y"]
tol = st.session_state["tol"]

# --- 左: プレビュー ---
with left:
    if frame is not None:
        st.session_state["_ankle_y"] = engine.lowest_ankle_y(people[t_idx]) if t_idx is not None else None
        img = engine.render_preview(frame, people, t_idx, ground_y, tol, start_f, None)
        st.image(img, **STRETCH)
        st.markdown(
            "🟧 **オレンジ** ＝ 追跡する走者　"
            "⬜ **グレー(IGNORE)** ＝ 無視する人　"
            "🟩 **緑の帯** ＝ 接地とみなす高さ　"
            "🔴 **赤丸** ＝ 帯に入った足首"
        )
        st.caption("足首が緑の帯に入った瞬間を「接地」として判定します。"
                   "走者の足が着地する高さに、帯の中心の線が来るように調整してください。")

# ----------------------------------------------------------------------
# ステップ4: 解析
# ----------------------------------------------------------------------
st.header("③ 解析する")
run = st.button("🚀 動作解析を開始する", type="primary", **STRETCH,
                disabled=(target_point is None))
if target_point is None:
    st.caption("走者が見つかると押せるようになります。")

if run:
    bar = st.progress(0.0)
    try:
        out_path, report, ex = engine.analyze_sprint(
            path, ground_y, start_f, last, target_point, tol, PREVIEW_WIDTH, dry_run=False,
            progress_callback=lambda f, t: bar.progress(f, text=t),
            strictness=STRICTNESS, max_steps=MAX_STEPS, model=MODEL_KIND)
        video_bytes = None
        if out_path and os.path.exists(out_path):
            with open(out_path, "rb") as fh:
                video_bytes = fh.read()
            os.remove(out_path)  # 即座に削除
        st.session_state["result"] = {"report": report, "extras": ex, "video": video_bytes}
    except Exception as e:
        st.error(f"解析中にエラーが発生しました: {e}")
    finally:
        bar.empty()
        gc.collect()

res = st.session_state.get("result")
if res:
    ex = res["extras"]
    st.divider()
    st.header("④ 結果")
    st.markdown(res["report"])
    st.download_button("📄 レポートをダウンロード (.md)", res["report"], file_name="sprint_report.md")

    if ex["captures"]:
        st.markdown("#### 📸 接地瞬間のキャプチャ")
        cols = st.columns(3)
        for i, (_, jpg, cap_text) in enumerate(ex["captures"]):
            cols[i % 3].image(jpg, caption=cap_text, **STRETCH)

    if ex.get("thumbs"):
        with st.expander("🔎 追跡確認（解析した人の切り抜き）— 同じ人が続いているか確認できます"):
            st.image([t[1] for t in ex["thumbs"]], caption=[f"F{t[0]}" for t in ex["thumbs"]], width=90)

    if res["video"]:
        st.markdown("#### 🎞️ 解析動画")
        if ex["playable"]:
            st.video(res["video"])
        else:
            st.caption("この環境のエンコーダーではブラウザ再生できない形式です。ダウンロードしてご確認ください。"
                       "（packages.txt に `ffmpeg` を追加するとブラウザで直接再生できます）")
        st.download_button("⬇️ 解析動画をダウンロード", res["video"], file_name="sprint_analysis.mp4",
                           mime="video/mp4")

st.divider()
st.button("🗑️ アップロード動画と解析データを破棄する", on_click=discard_and_reset_uploader)
