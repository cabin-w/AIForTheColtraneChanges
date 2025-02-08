import os
import librosa
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 音乐理论配置
CLUSTER_MAP = {
    # 五度圈顺时针排列（升号方向）
    0: {  # C Major
        "key": "C",
        "chords": [
            "Cmaj7",        # I级
            "G7alt",        # V7 (含b9/#9/b5/#5变化)
            "A7#11",        # 三全音替代D7 (II-V-I进行)
            "Ebmaj7",       # Coltrane变化关联和弦
            "F#dim7",       # 对称减和弦过渡
            "Bb7sus4"       # 三全音替代E7
        ],
        "function": "Tonic"
    },
    1: {  # G Major
        "key": "G",
        "chords": [
            "Gmaj7",
            "D7#9",         # 含布鲁斯音阶特征
            "B7alt",        # 三全音替代F7
            "Abmaj7",       # 来自Eb调关联
            "C#dim7",
            "F7b13"
        ],
        "function": "Dominant"
    },
    2: {  # D Major
        "key": "D",
        "chords": [
            "Dmaj7",
            "A7#5",
            "F7alt",        # 三全音替代B7
            "Bbmaj7",       # 远关系和弦
            "Ebdim7",
            "G7#11"
        ],
        "function": "Subdominant"
    },
    3: {  # A Major
        "key": "A",
        "chords": [
            "Amaj7",
            "E7+9",
            "C7alt",        # 三全音替代Gb7
            "Dbmaj7",       # Coltrane三调系统
            "Fdim7",
            "Bb7#5"
        ],
        "function": "Modulation Bridge"
    },
    4: {  # E Major
        "key": "E",
        "chords": [
            "Emaj7",
            "B7b9",
            "G7alt",        # 三全音替代Db7
            "Abmaj7",       # 三度关系调
            "Bbdim7",
            "D7#9"
        ],
        "function": "Tritone Shift"
    },
    5: {  # B Major (Coltrane主调1)
        "key": "B",
        "chords": [
            "Bmaj7",        # 主和弦
            "F#7alt",       # V7/III
            "Eb7#11",       # 三全音替代A7
            "Gmaj7",        # 关联调
            "C#dim7",       # 过渡和弦
            "D7b5"          # 德国增六和弦
        ],
        "function": "Primary Tonic"
    },
    6: {  # F# Major
        "key": "F#",
        "chords": [
            "F#maj7",
            "C#7+11",
            "A7alt",        # 三全音替代Eb7
            "Bbmaj7",       # 三全音关系
            "Ddim7",
            "E7#9"
        ],
        "function": "Pivot Modulation"
    },
    7: {  # C# Major (等价Db)
        "key": "Db",
        "chords": [
            "C#maj7",
            "G#7alt",
            "F7#11",        # 三全音替代B7
            "Emaj7",        # 远关系调
            "Gdim7",
            "A7b13"
        ],
        "function": "Enharmonic Gateway"
    },
    8: {  # G# Major (等价Ab)
        "key": "Ab",
        "chords": [
            "Abmaj7",
            "Eb7b9",
            "C7alt",        # 三全音替代F#7
            "Bmaj7",        # Coltrane三调关联
            "Dbdim7",
            "F7#5"
        ],
        "function": "Secondary Dominant"
    },
    9: {  # D# Major (等价Eb)
        "key": "Eb",
        "chords": [
            "Ebmaj7",       # Coltrane主调3
            "Bb7#9",
            "G7alt",        # 三全音替代C#7
            "Amaj7",        # 三度关系
            "Fdim7",
            "D7alt"         # 三全音替代Ab7
        ],
        "function": "Tertiary Tonic"
    },
    10: {  # A# Major (等价Bb)
        "key": "Bb",
        "chords": [
            "Bbmaj7",
            "F7alt",
            "D7#11",        # 三全音替代Ab7
            "Gbmaj7",       # 等音调
            "Adim7",
            "C7b9"
        ],
        "function": "Dominant Preparation"
    },
    11: {  # F Major
        "key": "F",
        "chords": [
            "Fmaj7",
            "C7+11",
            "A7alt",        # 三全音替代Eb7
            "Dmaj7",        # 三全音关联
            "G#dim7",
            "Bb7#9"
        ],
        "function": "Cycle Completion"
    }
}



def analyze_giant_steps(audio_path):
    try:
        # 1. 加载音频
        y, sr = librosa.load(audio_path, sr=44100, mono=True)
        if isinstance(y, np.ndarray):  # 确保加载的是 ndarray 类型
            print(f"✅ 音频加载成功 | 时长: {librosa.get_duration(y=y, sr=sr):.2f}秒")
        else:
            raise ValueError("音频加载失败，返回的不是 ndarray 类型")

        # 2. 节拍检测
        tempo, beat_frames = librosa.beat.beat_track(
            y=y, sr=sr, units='frames', tightness=100, trim=False
        )
        tempo = tempo if isinstance(tempo, float) else float(tempo[0])  # 确保tempo是一个浮动数字
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        print(f"🎶 节拍检测完成 | BPM: {tempo:.1f} | 节拍数: {len(beat_times)}")

        # 3. 和声分析
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, bins_per_octave=36, tuning=0)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)

        # 4. 增强特征提取
        chroma_mfcc_tonnetz = np.concatenate([chroma, mfcc, tonnetz], axis=0)

        # 5. 特征标准化
        scaler = StandardScaler()
        chroma_mfcc_tonnetz_norm = scaler.fit_transform(chroma_mfcc_tonnetz.T).T

        # 6. 使用 PCA 降维（有助于聚类的效果）
        pca = PCA(n_components=10)  # 降到 10 维
        chroma_mfcc_tonnetz_pca = pca.fit_transform(chroma_mfcc_tonnetz_norm.T)

        # 7. 聚类分析
        n_clusters = len(CLUSTER_MAP)  # 使用 CLUSTER_MAP 中的聚类数量
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        kmeans.fit(chroma_mfcc_tonnetz_pca)

        # 获取聚类标签
        labels = kmeans.labels_

        # 8. 生成时间轴数据
        changes = []
        for t in beat_times:
            # 获取聚类结果
            frame_idx = librosa.time_to_frames([t], sr=sr)[0]
            cluster = labels[frame_idx]

            # 获取当前和弦的调性和和弦
            cluster_info = CLUSTER_MAP.get(cluster, None)
            if cluster_info is None:
                continue

            current_key = cluster_info["key"]
            chord = cluster_info["chords"][0]  # 简化为取第一个和弦

            changes.append({
                "time": round(t, 2),
                "key": current_key,
                "chord": chord
            })

        return changes

    except Exception as e:
        print(f"❌ 分析失败: {str(e)}")
        return []


if __name__ == "__main__":
    AUDIO_FILE = "giant_steps.mp3"

    if not os.path.exists(AUDIO_FILE):
        print(f"⚠️ 错误：找不到音频文件 {AUDIO_FILE}")
        exit(1)

    result = analyze_giant_steps(AUDIO_FILE)

    if result:
        print("\n// 自动分析结果")
        print("CHORD_CHANGES = [")
        for c in result:
            print(f"  {{ time: {c['time']:.2f}, key: '{c['key']}', chord: '{c['chord']}' }},")
        print("];")
    else:
        print("⚠️ 未生成有效分析结果")
