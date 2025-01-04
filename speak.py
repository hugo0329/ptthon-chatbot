import speech_recognition as sr
import pyttsx3
from langchain_ollama import OllamaLLM  # 使用 OllamaLLM
from langchain.prompts import PromptTemplate

# 初始化文字轉語音引擎
engine = pyttsx3.init()

def speak(text):
    """將文字轉換為語音"""
    engine.say(text)
    engine.runAndWait()

def get_voice_input():
    """使用語音轉文字進行用戶輸入，無限重試直到成功"""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("請說話...")
        while True:  # 持續重試直到成功
            try:
                audio = recognizer.listen(source, timeout=10)
                text = recognizer.recognize_google(audio, language="zh-TW")
                print(f"您說了：{text}")
                return text
            except sr.UnknownValueError:
                speak("抱歉，我無法辨識您的語音。請再試一次。")
            except sr.RequestError:
                speak("語音服務出現問題。請稍後再試。")
                return None

def calculate_bmi(weight, height):
    """計算 BMI 並給出建議"""
    bmi = weight / (height / 100) ** 2
    if bmi < 18.5:
        return f"您的 BMI 是 {bmi:.1f}，屬於過輕範圍。建議增肌並增加健康飲食。"
    elif 18.5 <= bmi < 24.9:
        return f"您的 BMI 是 {bmi:.1f}，屬於正常範圍。可以進一步設定健身目標！"
    elif 25 <= bmi < 29.9:
        return f"您的 BMI 是 {bmi:.1f}，屬於過重範圍。建議進行減脂計畫並注意飲食控制。"
    else:
        return f"您的 BMI 是 {bmi:.1f}，屬於肥胖範圍。建議減脂計畫並與專家討論健康策略。"

def get_fitness_plan(goal, weight, height):
    """使用 OllamaLLM 生成個性化健身計畫"""
    llm = OllamaLLM(model="llama3.2")  # 使用 OllamaLLM
    template = """
    基於以下資訊生成個性化的健身計畫：
    - 體重：{weight} 公斤
    - 身高：{height} 公分
    - 健身目標：{goal}

    請給出專業建議，包括鍛煉計畫、推薦的動作，以及飲食建議。
    """
    prompt = PromptTemplate(input_variables=["goal", "weight", "height"], template=template)
    response = llm.invoke(prompt.format(goal=goal, weight=weight, height=height))  # 使用 `invoke` 方法
    return response

def main():
    """主程式邏輯"""
    speak("歡迎使用個性化健身計畫生成器。請提供您的體重和身高。")

    # 獲取體重
    speak("請告訴我您的體重，單位是公斤。")
    weight_input = get_voice_input()
    try:
        weight = float(weight_input)
    except ValueError:
        speak("無法辨識體重，請輸入數字。")
        return

    # 獲取身高
    speak("請告訴我您的身高，單位是公分。")
    height_input = get_voice_input()
    try:
        height = float(height_input)
    except ValueError:
        speak("無法辨識身高，請輸入數字。")
        return

    # 計算 BMI 並給建議
    bmi_suggestion = calculate_bmi(weight, height)
    speak(bmi_suggestion)
    print(bmi_suggestion)

    # 獲取健身目標
    speak("請告訴我您的健身目標，例如增肌、減脂或提升耐力。")
    goal = get_voice_input()

    # 生成健身計畫
    fitness_plan = get_fitness_plan(goal, weight, height)
    speak("根據您的需求，我為您生成了以下健身計畫：")
    print(fitness_plan)
    speak(fitness_plan)

if __name__ == "__main__":
    main()
