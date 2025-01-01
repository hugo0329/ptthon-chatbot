import ollama
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

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
            
            # 與 Ollama 交互
            response = self.ollama_client.generate(model="fit-coach", prompt=formatted_prompt)
            reply = response.get("content", "抱歉，我無法理解你的請求。")
            
            # 保存對話記憶
            self.memory.chat_memory.add_user_message(user_input)
            self.memory.chat_memory.add_ai_message(reply)
            
            return reply
        except Exception as e:
            return f"系統錯誤：{e}"
    
    def calculate_bmi(self, weight, height):
        """計算 BMI 並返回分類結果"""
        try:
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
            return f"計算錯誤：{e}"

# --- 主邏輯 ---
def main():
    print("你好！我是你的健身計畫聊天助手，隨時為你提供健身建議或生成健身計畫。")
    assistant = FitnessAssistant()
    
    while True:
        user_input = input("\n請輸入您的問題或需求（輸入 '退出' 結束）：\n")
        
        if user_input.strip() == "退出":
            print("感謝使用，再見！")
            break

        # 支援 BMI 計算的快捷功能
        if "BMI" in user_input.upper():
            try:
                weight = float(input("請輸入體重（公斤）："))
                height = float(input("請輸入身高（公尺）："))
                bmi_result = assistant.calculate_bmi(weight, height)
                print(f"\n助手：{bmi_result}")
            except ValueError:
                print("\n助手：請確保輸入的是正確的數字格式。")
            continue

        # 使用對話生成回應
        response = assistant.chat(user_input)
        print(f"\n助手：{response}")

# 啟動
if __name__ == "__main__":
    main()
