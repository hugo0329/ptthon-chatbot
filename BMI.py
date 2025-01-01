import ollama
import logging
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# 設定日誌
logging.basicConfig(level=logging.INFO)

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

# --- 健身助手類 ---
class FitnessAssistant:
    def __init__(self):
        self.ollama_client = ollama.Client()
        self.memory = memory
        self.prompt = prompt
        self.model_name = "llama2"  # 替換為已安裝的模型名稱

    def chat(self, user_input):
        """處理用戶輸入，返回生成的回應"""
        try:
            # 整理歷史記憶作為對話上下文
            history = "\n".join([
                f"用戶: {msg.content}" if msg.type == "human" else f"助手: {msg.content}"
                for msg in self.memory.chat_memory.messages
            ])
            
            # 格式化 Prompt
            formatted_prompt = self.prompt.format(history=history, input=user_input)
            logging.info(f"生成的 Prompt: {formatted_prompt}")
            
            # 嘗試使用模型生成回應
            response = self.ollama_client.generate(model=self.model_name, prompt=formatted_prompt)

            reply = response.get("content", "抱歉，我無法理解你的請求。")
            
            # 保存對話記憶
            self.memory.chat_memory.add_user_message(user_input)
            self.memory.chat_memory.add_ai_message(reply)
            
            return reply
        except Exception as e:
            logging.error(f"出現錯誤: {e}")
            return f"系統錯誤：{e}"
    
    def calculate_bmi(self, weight, height):
        """計算 BMI 並返回分類結果"""
        try:
            if weight <= 0 or height <= 0:
                return "體重和身高必須是正數，請重新輸入。"
            
            bmi = weight / (height ** 2)
            if bmi < 18.5:
                category = "體重過輕"
            elif 18.5 <= bmi < 24.9:
                category = "正常"
            elif 25 <= bmi < 29.9:
                category = "過重"
            else:
                category = "肥胖"
            return f"您的 BMI 是 {bmi:.2f}，屬於 {category} 範圍。"
        except ZeroDivisionError:
            return "身高不能為零，請重新輸入。"
        except Exception as e:
            logging.error(f"BMI 計算錯誤: {e}")
            return f"計算錯誤：{e}"
# --- 生成健身計畫函數 ---
def generate_fitness_plan(goal, current_fitness_level=None, available_time_per_week=None):
    """
    根據用戶的健身目標生成個性化健身計畫。
    """
    try:
        # 基本健身計畫模板
        if not current_fitness_level:
            current_fitness_level = "初學者"  # 默認為初學者
        if not available_time_per_week:
            available_time_per_week = 5  # 默認每週可鍛煉5小時

        plan = f"健身目標：{goal}\n目前健身水平：{current_fitness_level}\n每週可用鍛煉時間：約{available_time_per_week}小時\n\n"

        if goal == "增肌":
            plan += (
                "建議計畫：\n"
                "- 每週進行 4-5 次力量訓練，專注於大肌群（如胸、背、腿）。\n"
                "- 每組 6-12 次，3-4 組，重量選擇為 70%-85% 的最大負重。\n"
                "- 飲食建議：每天攝取比基礎代謝率多 300-500 卡路里，增加蛋白質（每公斤體重 1.6-2.2 克）。\n"
            )
        elif goal == "減脂":
            plan += (
                "建議計畫：\n"
                "- 每週進行 3 次力量訓練，2-3 次高強度間歇訓練（HIIT）或有氧運動。\n"
                "- 力量訓練每組 12-15 次，2-3 組，以中等重量為主。\n"
                "- 飲食建議：每天攝取比基礎代謝率少 300-500 卡路里，控制碳水化合物攝入，增加膳食纖維。\n"
            )
        elif goal == "耐力提升":
            plan += (
                "建議計畫：\n"
                "- 每週進行 3 次有氧運動（如跑步、游泳、騎行），每次 30-60 分鐘。\n"
                "- 1-2 次力量訓練，以全身性動作（如深蹲、硬舉、推舉）為主。\n"
                "- 運動強度逐步提高，可使用心率監測器，將心率控制在最大心率的 60%-80%。\n"
                "- 飲食建議：增加碳水化合物比例，確保訓練前後補充能量。\n"
            )
        else:
            plan += "抱歉，目前我們僅支持增肌、減脂或耐力提升的健身計畫建議。"

        return plan
    except Exception as e:
        return f"生成健身計畫時出現錯誤：{e}"

# --- 主邏輯 ---
def main():
    print("你好！我是你的健身計畫聊天助手，隨時為你提供健身建議或生成健身計畫。")
    assistant = FitnessAssistant()
    
    while True:
        user_input = input("\n請輸入您的問題或需求（輸入 '退出' 結束）：\n")
        
        if user_input.strip() == "退出":
            confirm = input("你確定要退出嗎？（輸入 '是' 確認，其他鍵取消）：")
            if confirm == "是":
                print("感謝使用，再見！")
                break
            continue
        if "BMI" in user_input.upper():
            try:
                weight = float(input("請輸入體重（公斤）："))
                height = float(input("請輸入身高（公尺）："))
                bmi_result = assistant.calculate_bmi(weight, height)
                print(f"\n助手：{bmi_result}")
            except ValueError:
                print("\n助手：請確保輸入的是正確的數字格式。")
            continue
        # 支援生成個性化健身計畫的功能
        if "健身計畫" in user_input:
            print("請選擇健身目標（增肌、減脂、耐力提升）：")
            goal = input("您的健身目標是：").strip()
            fitness_level = input("您的健身水平（初學者、中級、高級，可選）：").strip()
            time_per_week = input("您每週可用鍛煉時間（以小時計，可選）：").strip()
            
            try:
                fitness_level = fitness_level if fitness_level else None
                time_per_week = float(time_per_week) if time_per_week else None
                plan = generate_fitness_plan(goal, fitness_level, time_per_week)
                print(f"\n助手：以下是為您生成的健身計畫：\n{plan}")
            except ValueError:
                print("\n助手：請確保輸入的時間為數字格式。")
            continue

        # 使用對話生成回應
        response = assistant.chat(user_input)
        print(f"\n助手：{response}")

# --- 啟動 ---
if __name__ == "__main__":
    main()
