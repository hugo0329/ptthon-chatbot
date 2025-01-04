import speech_recognition as sr
import pyttsx3
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings

# 初始化文字轉語音引擎
engine = pyttsx3.init()

def speak(text):
    """將文字轉換為語音"""
    engine.say(text)
    engine.runAndWait()

def get_voice_input():
    """使用語音轉文字進行用戶輸入"""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("請說話...")
        try:
            audio = recognizer.listen(source, timeout=10)
            text = recognizer.recognize_google(audio, language="zh-TW")
            print(f"您說了：{text}")
            return text
        except sr.UnknownValueError:
            speak("抱歉，我無法辨識您的語音。請再試一次。")
            return None
        except sr.RequestError:
            speak("語音服務出現問題。")
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

# 初始化嵌入模型和向量檢索索引
embeddings_model = OllamaEmbeddings(model="llama3.2")
loader = TextLoader(file_path='./fitness_knowledge.txt', encoding="utf8")  # 背景知識檔案
index = VectorstoreIndexCreator(embedding=embeddings_model).from_loaders([loader])

def get_fitness_plan_with_rag(goal, weight, height):
    """使用 RAG 提供個性化健身計畫"""
    # 使用檢索獲取相關背景知識
    context = index.query(question=f"關於{goal}的健身建議，結合體重 {weight} 公斤和身高 {height} 公分的需求，提供背景資訊。")

    # 使用 Ollama 模型生成計畫
    llm = Ollama(model="llama3.2")
    template = """
    根據以下背景資訊和用戶資料，生成個性化的健身計畫：
    - 背景資訊：{context}
    - 體重：{weight} 公斤
    - 身高：{height} 公分
    - 健身目標：{goal}

    請給出專業建議，包括鍛煉計畫、推薦的動作，以及飲食建議。
    """
    prompt = PromptTemplate(input_variables=["context", "goal", "weight", "height"], template=template)
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(context=context, goal=goal, weight=weight, height=height)
    return response

# 主程式邏輯
def main():
    speak("歡迎使用個性化健身計畫生成器。請提供您的體重和身高。")

    # 獲取體重
    speak("請告訴我您的體重，單位是公斤。")
    weight_input = get_voice_input()
    if weight_input is None:
        return
    try:
        weight = float(weight_input)
    except ValueError:
        speak("無法辨識體重，請輸入數字。")
        return

    # 獲取身高
    speak("請告訴我您的身高，單位是公分。")
    height_input = get_voice_input()
    if height_input is None:
        return
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
    if goal is None:
        return

    # 使用 RAG 生成健身計畫
    fitness_plan = get_fitness_plan_with_rag(goal, weight, height)
    speak("根據您的需求，我為您生成了以下健身計畫：")
    speak(fitness_plan)
    print(fitness_plan)

# 運行主程式
if __name__ == "__main__":
    main()
