from  retrieval_component  import  RetrievalComponent,  deep_space_kb    
from  generative_component  import  GenerativeComponent    
    
class  HybridChatbot:    
        def  __init__(self):    
                print("正在初始化混合式聊天机器人...")    
                self.retriever  =  RetrievalComponent(deep_space_kb)    
                self.generator  =  GenerativeComponent()    
                print("聊天机器人已准备就绪！")    
    
        def  chat(self):    
                print("\n---  欢迎使用深空探测任务助手  ---")    
                print("您可以提问关于深空任务的问题，或者和我聊聊天。")    
                print("输入  '退出'  或  '再见'  来结束对话。")    
                print("输入  '清空历史'  来开始一段新的闲聊对话。")    
                    
                while  True:    
                        user_input  =  input("您:  ")    
                            
                        if  user_input.lower()  in  ['退出',  '再见']:    
                                print("机器人:  感谢您的使用，再见！")    
                                break    
                            
                        if  user_input.lower()  ==  '清空历史':    
                                self.generator.clear_history()    
                                print("机器人:  好的，我们的对话历史已清空。有什么新的话题吗？")    
                                continue    
    
                        #  1.  尝试使用检索组件    
                        retrieval_answer  =  self.retriever.get_answer(user_input,  similarity_threshold=0.7)    
                            
                        if  retrieval_answer:    
                                #  如果检索成功，直接输出准确答案    
                                print(f"机器人  (来自知识库):  {retrieval_answer}")    
                        else:    
                                #  2.  如果检索失败，使用生成组件进行对话    
                                print("机器人  (正在思考...):  ",  end="")    
                                generative_response  =  self.generator.generate_response(user_input)    
                                print(generative_response)    
    
if  __name__  ==  '__main__':    
        chatbot  =  HybridChatbot()    
        chatbot.chat()    