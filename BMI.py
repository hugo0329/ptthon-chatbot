import ollama
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# 初始化 Ollama 模型
ollama_client = ollama.Client()

# --- 健身計畫對話模板 ---
template = """
你是一個健身專家聊天機器人，能根據用戶的目標生成詳細的健身計畫，並回答健康相關問題。
以下是你需要提供的功能：
1. 根據 BMI 提供建議。
2. 根據健身目標（如增肌、減脂、耐力提升）生成個性化健身計畫。
3. 提供飲食建議或動作指導。

用戶的對話內容如下：
{history}

用戶剛剛的輸入是：
{input}

根據以上內容，請回答用戶的問題或提供建議。
"""
prompt = PromptTemplate(input_variables=["history", "input"], template=template)

# 初始化記憶
memory = ConversationBufferMemory()

# --- 功能函數 ---
def calculate_bmi(weight, height):
    """計算 BMI 並返回建議"""
    bmi = weight / (height ** 2)
    if bmi < 18.5:
        return f"你的 BMI 是 {bmi:.1f}，屬於體重過輕範圍，建議增加營養攝取並結合力量訓練。"
    elif 18.5 <= bmi < 24.9:
        return f"你的 BMI 是 {bmi:.1f}，屬於健康範圍，請繼續保持健康的生活方式！"
    elif 25 <= bmi < 29.9:
        return f"你的 BMI 是 {bmi:.1f}，屬於體重過重範圍，建議控制飲食並增加有氧運動。"
    else:
        return f"你的 BMI 是 {bmi:.1f}，屬於肥胖範圍，建議在專業人士指導下進行飲食控制和系統健身計畫。"

def generate_fitness_plan(goal):
    """根據健身目標生成個性化計畫"""
    prompt = f"你是一個健身教練，幫助用戶達成健身目標。用戶的目標是：{goal}。請生成一個詳細的健身計畫。"
    response = ollama_client.generate(model="llama2", prompt=prompt)
    generated_plan = response.get("content", "").strip()
    
    if not generated_plan:
        # 提供預設建議
        if goal.lower() == "增肌":
            generated_plan = (
                "增肌計畫建議：\n"
                "- 每週進行 4-5 次力量訓練，專注於多關節動作如深蹲、硬拉和臥推。\n"
                "- 每天攝取的蛋白質量為每公斤體重 1.6-2.2 克，碳水化合物和健康脂肪比例保持平衡。\n"
                "- 確保每晚睡眠 7-9 小時，促進肌肉修復與成長。"
            )
        elif goal.lower() == "減脂":
            generated_plan = (
                "減脂計畫建議：\n"
                "- 每週進行 3-5 次有氧運動，如跑步或高強度間歇訓練（HIIT）。\n"
                "- 每天攝取的熱量控制在總消耗量以下，保持高蛋白飲食以保留肌肉質量。\n"
                "- 結合每週 2-3 次力量訓練，提升基礎代謝率。"
            )
        elif goal.lower() == "耐力提升":
            generated_plan = (
                "耐力提升計畫建議：\n"
                "- 每週進行 4-6 次心肺耐力訓練，如長跑、游泳或騎行。\n"
                "- 間歇加入短跑或高強度運動，提高心肺能力和速度。\n"
                "- 保持充足的碳水化合物攝取，確保訓練過程中有足夠能量。\n"
                "- 每週至少進行一次休息或輕度活動，防止過度訓練。"
            )
        else:
            generated_plan = "目前僅支持增肌、減脂和耐力提升的建議計畫，請輸入其中之一的目標！"
    
    return generated_plan


def provide_diet_or_exercise_advice(topic):
    """提供飲食或動作指導"""
    prompt = f"你是一個健身專家。請提供有關 {topic} 的專業建議。"
    response = ollama_client.generate(model="llama2", prompt=prompt)
    generated_advice = response.get("content", "").strip()
    
    if not generated_advice:
        # 提供預設建議
        if "飲食" in topic:
            generated_advice = (
                "一般飲食建議：\n"
                "- 保持飲食均衡，包含足量的蛋白質、碳水化合物和健康脂肪。\n"
                "- 多攝取蔬菜和水果，補充必要的維生素與礦物質。\n"
                "- 控制加工食品和含糖飲料的攝取，選擇天然食材。\n"
                "- 根據目標調整熱量攝取（減脂需熱量赤字，增肌需熱量盈餘）。"
            )
        elif "動作" in topic or "運動" in topic:
            generated_advice = (
                "一般動作指導建議：\n"
                "- 力量訓練時確保動作正確，避免不必要的受傷風險。\n"
                "- 使用適當的重量，確保能完成每組 8-12 次。\n"
                "- 有氧運動如跑步時，保持穩定的呼吸和適當的心率區間。\n"
                "- 訓練後進行充分的拉伸，促進肌肉放鬆與恢復。"
            )
        else:
            generated_advice = "目前僅支持飲食和動作相關的建議，請指定更清晰的主題！"
    
    return generated_advice

# --- 主邏輯 ---
def main():
    print("你好！我是你的健身計畫聊天助手，隨時為你提供健身建議或生成健身計畫。")
    
    while True:
        user_input = input("\n請輸入您的問題或需求（如 '計算 BMI' 或 '生成健身計畫'，輸入 '退出' 結束）：\n")
        
        if user_input.strip() == "退出":
            print("感謝使用，再見！")
            break
        
        if user_input.strip().lower().startswith("bmi"):
            try:
                weight = float(input("請輸入你的體重（公斤）："))
                height = float(input("請輸入你的身高（米）："))
                bmi_result = calculate_bmi(weight, height)
                print(f"\n助手：{bmi_result}")
            except ValueError:
                print("\n助手：請輸入有效的數字！")
        
        elif user_input.strip().lower().startswith("健身計畫"):
            goal = input("請輸入你的健身目標（如增肌、減脂、耐力提升）：")
            fitness_plan = generate_fitness_plan(goal)
            print(f"\n助手：{fitness_plan}")
        
        elif user_input.strip().lower().startswith("建議"):
            topic = input("請描述你需要的建議（如飲食、特定運動）：")
            advice = provide_diet_or_exercise_advice(topic)
            print(f"\n助手：{advice}")
        
        else:
            print("\n助手：抱歉，我無法理解你的請求，請嘗試輸入 '計算 BMI' 或 '生成健身計畫' 等關鍵字。")

# 啟動
if __name__ == "__main__":
    main()
