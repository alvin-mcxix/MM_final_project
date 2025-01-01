from keras.models import load_model # 執行 Keras 所需 Tensorflow==2.14
import numpy as np                  # numpy==1.26.4
import cv2

from pydub import AudioSegment
from pydub.playback import _play_with_simpleaudio

# 輸入音符的播放速度（每次會播 一個4分音符 長）
bpm = int(input("BPM = "))


# 文字設定
org = (75, 120)                         # 位置
fontFace = cv2.FONT_HERSHEY_SIMPLEX     # 字型
fontScale = 2.5                         # 尺寸
color = (0, 0, 0)                       # 顏色
thickness = 5                           # 外框線條粗細
lineType = cv2.LINE_AA                  # 外框線條樣式

# 載入已經訓練好的模型
model = load_model("keras_model.h5", compile=False)

# 定義分類名稱
CLASS_NAME = ["Empty", "Do", "Re", "Mi", "Fa", "Sol", "La", "Si", "hDo", "Stop"]


# 定義對應鋼琴音符的頻率（以 Hz 為單位）
NOTES = {
    "Do":   261.63, # C4
    "Re":   293.66, # D4
    "Mi":   329.63, # E4
    "Fa":   349.23, # F4
    "Sol":  392.00, # G4
    "La":   440.00, # A4
    "Si":   493.88, # B4
    "hDo":  523.25  # C5
}

# 生成一個具有指定頻率和持續時間的音調，並以 AudioSegment 的形式返回
def generate_tone(frequency, duration, sample_rate=44100, amplitude=0.5):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = (amplitude * np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)
    audio_segment = AudioSegment(
        wave.tobytes(),
        frame_rate=sample_rate,
        sample_width=wave.dtype.itemsize,
        channels=1
    )
    return audio_segment

# 全域變數，用於控制音符的播放
current_playback = None # 保存目前播放的音訊對象
playing_note = None     # 保存目前播放的音符名稱

def play_note(note, duration=60/bpm):
    global current_playback, playing_note

    # 如果目前播放的音符與預測結果相同，且仍在播放，則忽略
    if note == playing_note and current_playback is not None and current_playback.is_playing():
        return

    # 停止目前正在播放的音符
    if current_playback is not None:
        current_playback.stop()

    # 播放新音符
    if note != "Empty" and note != "Stop":
        frequency = NOTES[note]
        tone = generate_tone(frequency, duration)
        current_playback = _play_with_simpleaudio(tone)  # 非阻塞播放
        playing_note = note  # 更新目前播放的音符

# 開啟相機
camera = cv2.VideoCapture(0)
width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))   # 取得影像寬度
height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)) # 取得影像高度

# 讀取遮罩圖片，並調整為相機畫面比例
mask = cv2.resize(cv2.imread('mask.png'), (width, height))

# 將遮罩轉為灰階圖像，並提取黑色區域
mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
_, mask_binary = cv2.threshold(mask_gray, 1, 255, cv2.THRESH_BINARY_INV)

# 計算黑色區域的邊界框
coords = cv2.findNonZero(mask_binary)  # 獲取黑色像素的座標
x, y, w, h = cv2.boundingRect(coords)  # 計算黑色區域的邊界框
print(f"Cropping region: x={x}, y={y}, w={w}, h={h}")


# 讀取顯示用的背景圖片，並調整為相機畫面比例
square = cv2.resize(cv2.imread('square.png'), (width, height))

while True:
    # 從鏡頭截取畫面
    ret, frame = camera.read()
    
    # 裁剪圖像（即 mask 的黑色範圍）
    cropped_frame = frame[y:y+h, x:x+w]
    
    # 將圖像調整為模型需要的大小 (224 x 224 像素)
    cropped_frame = cv2.resize(cropped_frame, (224, 224), interpolation=cv2.INTER_AREA)
    
    # 將圖像轉為 NumPy 陣列並調整形狀
    cropped_frame = np.asarray(cropped_frame, dtype=np.float32).reshape(1, 224, 224, 3)

    # 正規化圖像數據
    cropped_frame = (cropped_frame / 127.5) - 1

    # 使用模型進行預測
    prediction = model.predict(cropped_frame, verbose=0)
    index = np.argmax(prediction)
    note = CLASS_NAME[index]  # 獲取預測的音符名稱
    confidence_score = prediction[0][index]  # 獲取信賴指數
    
    # 播放音符
    play_note(note)
    
    # 將鏡頭截取的畫面左右翻轉（因為使用電腦前鏡頭，這樣能使畫面變鏡像）並疊圖一張方形圖（表示手勢偵測區）
    show_frame = cv2.addWeighted(cv2.flip(frame, 1), 1, square, 0.1, 0)
    
    # 在畫面加入文字
    text = f"{note}:{str(np.round(confidence_score * 100))[:-2]}%"
    cv2.putText(show_frame, text, org, fontFace, fontScale, color, thickness, lineType)
    
    # 在視窗中顯示結果
    cv2.imshow("Camera Image", show_frame)

    # 鍵盤監測：如果按下 ESC 鍵（ASCII 27），退出程式
    keyboard_input = cv2.waitKey(1)
    if keyboard_input == 27:
        break

# 釋放鏡頭並關閉視窗
camera.release()
cv2.destroyAllWindows()
