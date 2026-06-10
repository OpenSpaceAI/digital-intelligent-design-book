from transformers import AutoModelForCausalLM,AutoTokenizer
import torch

class GenerativeComponent:
    def __init__(self, model_name="microsoft/DialoGPT-small"):
        """
        初始化生成组件
        :param model_name: Hugging Face上的预训练对话模型名称
        """
        print("正在加载生成式对话模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.chat_history_ids = None
        print("模型加载完成。")

def generate_response(self, user_input):
    """
    根据用户输入生成对话回复
    :param user_input: 用户的输入字符串
    :return: 模型生成的回复字符串
    """
    # 将用户输入编码为模型可接受的格式，并附加结束符
    new_user_input_ids = self.tokenizer.encode(user_input + self.tokenizer.eos_token, return_tensors='pt')

    # 将当前用户输入拼接到对话历史中
    if self.chat_history_ids is not None:
        bot_input_ids = torch.cat([self.chat_history_ids, new_user_input_ids], dim=-1)
    else:
        bot_input_ids = new_user_input_ids

    # 生成回复，并更新对话历史
    # max_length 控制对话历史的总长度，避免无限增长
    # pad_token_id 确保生成过程中的填充正确
    self.chat_history_ids = self.model.generate(
    bot_input_ids,
    max_length=1000,
    pad_token_id=self.tokenizer.eos_token_id,
    no_repeat_ngram_size=3, # 避免重复的短语
    do_sample=True, # 使用采样以增加多样性
    top_k=50, # 限制采样范围
    top_p=0.95, # 使用核心采样
    temperature=0.7 # 控制生成的随机性
    )

    # 解码生成的回复，并只返回最新一轮的机器人回答
    response = self.tokenizer.decode(self.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response

def clear_history(self):
    """清空对话历史"""
    self.chat_history_ids = None

    if __name__ == '__main__':
    # 测试生成组件
        generator = GenerativeComponent()

        response1 = generator.generate_response("你好，你叫什么名字？")
        print("AI:", response1)

        response2 = generator.generate_response("你对太空探索感兴趣吗？")
        print("AI:", response2)