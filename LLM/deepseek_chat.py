from langchain.chat_models import init_chat_model

from util.env_util import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL

deepseek_llm = init_chat_model(
    model = "deepseek-chat",
    model_provider = "deepseek",
    api_key =DEEPSEEK_API_KEY,
    base_url = DEEPSEEK_BASE_URL
)


if __name__ == "__main__":
    response = deepseek_llm.invoke("你好,简单解释一下你自己")
    print(type(response))
    print(response.content)