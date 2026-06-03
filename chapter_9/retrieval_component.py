import   numpy   as   np      
from   sentence_transformers   import   SentenceTransformer      
from   sklearn.metrics.pairwise   import   cosine_similarity      
      
class   RetrievalComponent:      
            def   __init__(self,   knowledge_base):      
                        """   
                        初始化检索组件   
                        :param   knowledge_base:   一个字典，键是问题，值是答案   
                        """      
                        self.knowledge_base   =   knowledge_base      
                        self.questions   =   list(self.knowledge_base.keys())      
                        self.answers   =   list(self.knowledge_base.values())      
                              
                        #   加载预训练的句子向量模型，这个模型擅长将句子映射到有意义的向量空间      
                        #   'all-MiniLM-L6-v2'   是一个轻量且高效的模型      
                        print("正在加载句子向量模型...")      
                        self.model   =   SentenceTransformer('all-MiniLM-L6-v2')      
                        print("模型加载完成。正在编码知识库...")      
                              
                        #   对知识库中的所有问题进行编码，生成向量表示，并存储起来      
                        self.question_embeddings   =   self.model.encode(self.questions)      
                        print("知识库编码完成。")      
      
            def   get_answer(self,   user_query,   similarity_threshold=0.7):      
                        """   
                        根据用户查询，从知识库中检索答案   
                        :param   user_query:   用户的输入字符串   
                        :param   similarity_threshold:   相似度阈值，低于此阈值则认为没有找到匹配项   
                        :return:   如果找到匹配答案则返回答案字符串，否则返回None   
                        """      
                        #   将用户的查询也编码成向量      
                        query_embedding   =   self.model.encode([user_query])      
                              
                        #   计算用户查询向量与知识库中所有问题向量的余弦相似度      
                        similarities   =   cosine_similarity(query_embedding,   self.question_embeddings)      
                              
                        #   找到最相似的问题的索引和相似度值      
                        max_similarity_index   =   np.argmax(similarities)      
                        max_similarity_value   =   similarities[0,   max_similarity_index]      
                              
                        print(f"与最相似问题   '{self.questions[max_similarity_index]}'   的相似度为:   {max_similarity_value:.4f}")      
                              
                        #   如果最高相似度超过设定的阈值，则认为找到了匹配的答案      
                        if   max_similarity_value   >=   similarity_threshold:      
                                    return   self.answers[max_similarity_index]      
                        else:      
                                    return   None      
      
#   定义我们的深空探测知识库      
deep_space_kb   =   {      
            "旅行者1号的主要任务是什么？":   "旅行者1号的主要任务是探测木星和土星及其卫星。完成行星探测后，它的新任务是探索太阳系的边缘，并最终进入星际空间。",      
            "毅力号火星车在哪里着陆的？":   "毅力号火星车于2021年2月18日在火星的耶泽罗陨石坑（Jezero   Crater）成功着陆。",      
            "詹姆斯·韦伯空间望远镜的目标是什么？":   "詹姆斯·韦伯空间望远镜的主要目标是观测宇宙中最古老的恒星和星系，研究恒星与行星系统的形成，并探索系外行星的潜在生命迹象。",      
            "阿尔忒弥斯计划的目标是什么？":   "阿尔忒弥斯计划是美国宇航局（NASA）的一项载人航天计划，目标是重返月球并建立可持续的人类存在，并为未来的火星任务做准备。",      
            "新视野号探测了哪个天体？":   "新视野号（New   Horizons）的主要探测目标是冥王星及其卫星。在飞越冥王星后，它继续探测了柯伊伯带天体“天空”（Arrokoth）。",      
            "介绍一下哈勃望远镜。":   "哈勃空间望远镜是在地球轨道上运行的空间望远镜，自1990年以来，它提供了大量关于宇宙的珍贵图像和数据，极大地加深了我们对宇宙的理解。"      
}      
      
if   __name__   ==   '__main__':      
            #   测试检索组件      
            retriever   =   RetrievalComponent(deep_space_kb)      
            test_query_1   =   "韦伯望远镜用来干什么的？"      
            answer_1   =   retriever.get_answer(test_query_1)      
            print(f"用户问题:   {test_query_1}")      
            print(f"机器人回答:   {answer_1}\n")      
            test_query_2   =   "今天天气怎么样？"      
            answer_2   =   retriever.get_answer(test_query_2)      
            print(f"用户问题:   {test_query_2}")      
            print(f"机器人回答:   {answer_2}")      