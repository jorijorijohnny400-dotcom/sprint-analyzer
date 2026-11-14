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

st.title("🏃‍♂️ スプリント バイオメカニクス 診断システム")
st.markdown("横から撮影したスプリント動画をアップロードし、ラルフ・マンのエリート基準に基づいた動作解析を行います。")

PREVIEW_WIDTH = 640


# ----------------------------------------------------------------------
# キャッシュ付きフレーム取得 + 人物検出 (プレビュー用)
# ----------------------------------------------------------------------
@st.cache_data(show_spinner=False, max_entries=24)
def cached_frame_people(path, idx, kind="full"):
    frame = engine.read_frame(path, idx, PREVIEW_WIDTH)
    if frame is None:
        return None, []
    return frame, engine.detect_people_image(frame, kind)


def discard_video():
    p = st.session_state.get("video_path")
    if p and os.path.exists(p):
        os.remove(p)
    for k in ("video_path", "video_id", "info", "result", "dry"):
        st.session_state.pop(k, None)
    cached_frame_people.clear()
    gc.collect()


# ----------------------------------------------------------------------
# アップロード
# ----------------------------------------------------------------------
st.subheader("📹 1. スプリント動画のアップロード")
uploaded = st.file_uploader("スプリント動画を選択してください (.mp4, .mov)", type=["mp4", "mov"])

if uploaded is None:
    if "video_path" in st.session_state:
        discard_video()
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
    for k in ("range", "ground_y", "pv_frame", "target_no"):
        st.session_state.pop(k, None)

path = st.session_state["video_path"]
info = st.session_state["info"]
n_frames = info["frames"]
fps = info["fps"]
last = n_frames - 1

st.caption(f"{info['width']}×{info['height']} / {fps:.1f}fps / {n_frames}フレーム "
           f"（解析時は横幅{PREVIEW_WIDTH}pxに自動縮小します）")

# 初期値
st.session_state.setdefault("range", (0, last))
st.session_state.setdefault("ground_y", 0.80)
st.session_state.setdefault("tol", 0.05)


# ----------------------------------------------------------------------
# コールバック
# ----------------------------------------------------------------------
def cb_find_first():
    with st.spinner("最初に人が映るフレームを探しています..."):
        _k = "full" if st.session_state.get("model_kind", "高精度 (full)").startswith("高精度") else "lite"
        f = engine.find_first_person_frame(path, 0, kind=_k)
    if f is None:
        st.session_state["_msg"] = "人物を検出できませんでした。"
    else:
        s, e = st.session_state["range"]
        st.session_state["range"] = (f, max(f + 1, e))
        st.session_state["_msg"] = f"フレーム {f} で最初の人物を検出しました。開始位置に設定しました。"


def cb_fit_ground():
    y = st.session_state.get("_ankle_y")
    if y is not None:
        st.session_state["ground_y"] = float(min(0.99, max(0.30, round(y, 3))))


def cb_set_end_from_lock():
    d = st.session_state.get("dry")
    if d and d["extras"]["lock_end"] is not None:
        s, _ = st.session_state["range"]
        st.session_state["range"] = (s, int(d["extras"]["lock_end"]))


# ----------------------------------------------------------------------
# プレビュー & 調整
# ----------------------------------------------------------------------
st.subheader("🎯 2. プレビューで走者・接地ライン・解析範囲を調整")
if st.session_state.get("_msg"):
    st.info(st.session_state.pop("_msg"))

st.radio("姿勢推定モデル（精度が低いと感じたら『高精度』）", ["高精度 (full)", "軽量 (lite)"],
         horizontal=True, key="model_kind")
kind = "full" if st.session_state["model_kind"].startswith("高精度") else "lite"

left, right = st.columns([3, 2])

with right:
    st.markdown("**① 解析範囲（開始〜終了フレーム）**")
    st.slider("解析するフレーム範囲", 0, last, key="range")
    st.button("🔍 最初に人が映るフレームを自動検出", on_click=cb_find_first)
    start_f, end_f = st.session_state["range"]
    st.caption(f"範囲: {start_f}〜{end_f}（{start_f / fps:.2f}秒〜{end_f / fps:.2f}秒）")

    st.markdown("**② プレビューで表示するフレーム**")
    show = st.radio("表示", ["開始フレーム", "任意のフレーム", "終了フレーム"], horizontal=True,
                    label_visibility="collapsed")
    if show == "開始フレーム":
        pv = start_f
    elif show == "終了フレーム":
        pv = end_f
    else:
        pv = st.slider("フレーム位置", 0, last, value=min(max(start_f, 0), last), key="pv_frame")

# 開始フレームの人物検出 → ターゲット選択
start_frame_img, start_people = cached_frame_people(path, start_f, kind)

with right:
    st.markdown("**③ ターゲット選手（最初に追跡する人）**")
    target_point = None
    target_idx_start = None
    if not start_people:
        st.warning("開始フレームで人物が検出されません。開始位置を調整するか、自動検出ボタンを使ってください。")
    else:
        default_i = engine.default_target_index(start_people)
        labels = [f"#{i + 1}（画面左から{i + 1}人目）" for i in range(len(start_people))]
        if len(start_people) > 1:
            sel = st.radio("開始フレームに複数人が映っています", labels, index=default_i)
            target_idx_start = labels.index(sel)
        else:
            target_idx_start = 0
            st.success("開始フレームの走者 1人をロックオンします。")
        target_point = engine.person_center(start_people[target_idx_start])

    st.select_slider(
        "追跡の厳しさ（別の人に乗り移る → 厳しい ／ すぐ追跡が切れる → 緩い）",
        options=["緩い", "標準", "厳しい"], value="標準", key="strict")

    st.markdown("**⑤ 解析する歩数**")
    unlimited = st.checkbox("歩数を制限しない（ロックした人がフレームアウトするまで）", value=False, key="unlimited")
    st.slider("ロックした人の最初のN歩だけ解析する", 1, 12, 3, key="n_steps", disabled=unlimited)
    max_steps = 0 if unlimited else int(st.session_state["n_steps"])
    if not unlimited:
        st.caption(f"最初の{max_steps}歩の接地を取得した時点で解析を終了します（後半の誤検出や乗り移りを防ぎます）。")

    st.markdown("**④ 接地ライン**")
    st.slider("地面の高さ（上=0 / 下=1）", 0.30, 0.99, step=0.005, key="ground_y")
    st.slider("接地ゾーンの幅（広いほど接地を拾いやすい）", 0.02, 0.15, step=0.005, key="tol")
    st.button("👟 ターゲットの足首の高さに自動で合わせる（表示中のフレーム）", on_click=cb_fit_ground)

ground_y = st.session_state["ground_y"]
tol = st.session_state["tol"]
# 追跡結果が有効かどうかの判定用（範囲・ターゲットが変わったら無効）
current_params = (
    start_f, end_f,
    tuple(round(float(v), 3) for v in target_point) if target_point is not None else None,
    st.session_state.get("strict", "標準"),
    max_steps, kind,
)

with left:
    frame, people = cached_frame_people(path, pv, kind)
    if frame is None:
        st.error("このフレームを読み込めませんでした。")
    else:
        _dry = st.session_state.get("dry")
        track_ok = bool(_dry) and _dry.get("params") == current_params
        roles = True
        note = ""
        if pv == start_f:
            # 開始フレーム: 選択した人がターゲット
            t_idx = engine.pick_nearest(people, target_point) if people else None
        elif track_ok:
            # 追跡テスト済み: 実際にロックされていた位置の人だけがターゲット
            c = _dry["extras"]["track"].get(pv)
            t_idx = engine.match_center(people, c) if c is not None else None
            if t_idx is None:
                note = "このフレームではターゲット未追跡（他の人はすべて無視されます）"
        else:
            t_idx = None
            roles = False
            note = "追跡テストを実行すると、このフレームでロックされている人が表示されます"
        st.session_state["_ankle_y"] = engine.lowest_ankle_y(people[t_idx]) if t_idx is not None else None
        in_range = start_f <= pv <= end_f
        img = engine.render_preview(frame, people, t_idx, ground_y, tol, pv, in_range, banner="", roles=roles)
        st.image(img, use_container_width=True)
        st.caption(
            "🟧 オレンジ = ロックオン中のターゲット（1人だけ）　⬜ グレー(IGNORE) = 無視される人　"
            "🟢 緑ライン/帯 = 接地判定ゾーン　🔴 赤丸 = 接地ゾーン内の足首　"
            "枠の色：緑=解析範囲内 / 赤=範囲外"
        )
        if note:
            st.info(note)

# ----------------------------------------------------------------------
# 追跡テスト (動画出力なし)
# ----------------------------------------------------------------------
st.subheader("🧪 3. 追跡テスト（任意・動画出力なし）")
st.caption("ターゲットが最後まで追跡できるか、接地が検出されるかを先に確認できます。")

if st.button("追跡テストを実行"):
    bar = st.progress(0.0)
    try:
        _, rep, ex = engine.analyze_sprint(
            path, ground_y, start_f, end_f, target_point, tol, PREVIEW_WIDTH, dry_run=True,
            progress_callback=lambda f, t: bar.progress(f, text=t),
            strictness=st.session_state.get("strict", "標準"), max_steps=max_steps, model=kind)
        st.session_state["dry"] = {"report": rep, "extras": ex, "ground": ground_y, "tol": tol,
                                   "params": current_params}
    except Exception as e:
        st.error(f"追跡テストでエラー: {e}")
    finally:
        bar.empty()
        gc.collect()

dry = st.session_state.get("dry")
if dry:
    ex = dry["extras"]
    c1, c2 = st.columns([2, 1])
    with c2:
        st.metric("検出された接地数", len(ex["contacts"]))
        if ex["lock_start"] is not None:
            st.write(f"ロック範囲: Frame {ex['lock_start']}〜{ex['lock_end']}")
        st.write(ex["end_reason"])
        if ex["lock_end"] is not None:
            st.button("終了フレームをロック終了位置に設定", on_click=cb_set_end_from_lock)
        if (dry["ground"], dry["tol"]) != (ground_y, tol):
            st.warning("接地ラインを変更しています。再度テストしてください。")
    with c1:
        import pandas as pd

        rows = [{"frame": f, "足首の高さ": (None if y is None else 1 - y)} for f, y in ex["trajectory"]]
        df = pd.DataFrame(rows).set_index("frame")
        df["接地ライン"] = 1 - dry["ground"]
        df["接地ゾーン上端"] = 1 - (dry["ground"] - dry["tol"])
        st.line_chart(df)
        st.caption("足首の高さが『接地ゾーン上端』を下回った瞬間が接地です。途切れた部分はターゲット未検出です。"
                   + (f" 検出した接地フレーム: {ex['contacts']}" if ex["contacts"] else ""))
    if ex.get("thumbs"):
        st.markdown("**🔎 追跡確認：ロックしている人の切り抜き（同じ人が続いているか確認してください）**")
        st.image([t[1] for t in ex["thumbs"]], caption=[f"F{t[0]}" for t in ex["thumbs"]], width=90)
        st.caption("別の人が混ざっている場合は「追跡の厳しさ」を『厳しい』に、途中で切れる場合は『緩い』にして再テストしてください。")

# ----------------------------------------------------------------------
# 本解析
# ----------------------------------------------------------------------
st.subheader("🚀 4. 動作解析の実行")
if st.button("動作解析を開始する", type="primary"):
    if target_point is None:
        st.error("開始フレームで人物が検出されていません。範囲を調整してください。")
    else:
        bar = st.progress(0.0)
        try:
            out_path, report, ex = engine.analyze_sprint(
                path, ground_y, start_f, end_f, target_point, tol, PREVIEW_WIDTH, dry_run=False,
                progress_callback=lambda f, t: bar.progress(f, text=t),
                strictness=st.session_state.get("strict", "標準"), max_steps=max_steps, model=kind)
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
    st.markdown(res["report"])
    st.download_button("📄 レポートをダウンロード (.md)", res["report"], file_name="sprint_report.md")

    if ex["captures"]:
        st.markdown("#### 📸 接地瞬間のキャプチャ")
        cols = st.columns(3)
        for i, (_, jpg, cap_text) in enumerate(ex["captures"]):
            cols[i % 3].image(jpg, caption=cap_text, use_container_width=True)

    if ex.get("thumbs"):
        st.markdown("#### 🔎 追跡確認（解析した人の切り抜き）")
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
st.button("🗑️ アップロード動画と解析データを破棄する", on_click=discard_video)
