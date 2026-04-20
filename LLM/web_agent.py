import os
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from langchain.agents import create_agent
from langchain_core.tools import tool
from LLM.deepseek_chat import deepseek_llm

os.environ['USER_AGENT'] = 'WebAgent/1.0'


@tool
async def get_webpage_summary(url: str) -> str:
    """传入URL，抓取网页正文，生成摘要"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers={"User-Agent": os.environ['USER_AGENT']}) as response:
                response.raise_for_status()
                html_content = await response.text()

        # 使用BeautifulSoup解析HTML，提取文本
        soup = BeautifulSoup(html_content, 'html.parser')
        # 提取正文文本，并清理空白字符
        text = soup.get_text(separator='\n', strip=True)

        if not text:
            return "错误：未能从网页提取到任何内容。"

        content = text[:6000]
        return f"网页正文：{content}\n请精简总结核心摘要"

    except Exception as e:
        return f"网页抓取失败：{str(e)}"


deepseek_agent = create_agent(
    model=deepseek_llm,
    tools=[get_webpage_summary],
    system_prompt="你是一个网页摘要助手，可以帮助用户总结传入的url的内容，使用一句话总结网页内容"
)


async def main(url: str):
    result = await deepseek_agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": f"请帮我总结一下这个网页的内容：{url}"
                }
            ]
        }
    )

    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main("https://blog.csdn.net/weixin_60925698/article/details/159695677?spm=1000.2115.3001.10524"))