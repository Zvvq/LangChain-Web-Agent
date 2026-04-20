import os
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from langchain.agents import create_agent
from langchain_core.tools import tool
from LLM.deepseek_chat import deepseek_llm

USER_AGENT = "MyWebAgent/1.0"
MAX_HTML_SIZE = 2 * 1024 * 1024   # 2MB
TIMEOUT_SECONDS = 10
CHUNK_SIZE = 8192

@tool
async def get_webpage_summary(url: str) -> str:
    """传入URL，抓取网页正文，生成摘要"""
    try:
        timeout = aiohttp.ClientTimeout(total=TIMEOUT_SECONDS)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, headers={"User-Agent": USER_AGENT}) as response:
                response.raise_for_status()

                # 流式读取，限制大小
                chunks = []
                total_size = 0
                async for chunk in response.content.iter_chunked(CHUNK_SIZE):
                    total_size += len(chunk)
                    if total_size > MAX_HTML_SIZE:
                        keep_size = MAX_HTML_SIZE - (total_size - len(chunk))
                        if keep_size > 0:
                            chunks.append(chunk[:keep_size])
                        break
                    chunks.append(chunk)

                html_bytes = b''.join(chunks)
                encoding = response.get_encoding() or 'utf-8'
                html_content = html_bytes.decode(encoding, errors='ignore')

        # 使用 lxml 解析器
        soup = BeautifulSoup(html_content, 'lxml')

        # === 通用清理：移除所有常见非内容标签 ===
        noise_tags = [
            'script', 'style', 'nav', 'header', 'footer', 'aside',
            'form', 'iframe', 'noscript', 'link', 'meta', 'button'
        ]
        for tag in soup(noise_tags):
            tag.decompose()

        # 提取纯文本
        text = soup.get_text(separator='\n', strip=True)

        if not text:
            return "错误：未能从网页提取到任何内容。"

        content = text[:6000]
        return f"网页正文（前6000字符）：{content}\n请基于以上内容生成摘要。"

    except asyncio.TimeoutError:
        return "网页抓取失败：请求超时，网页可能过大或网络不稳定。"
    except aiohttp.ClientError as e:
        return f"网页抓取失败：网络请求错误 - {str(e)}"
    except Exception as e:
        return f"网页抓取失败：未知错误 - {str(e)}"


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
    asyncio.run(main("https://blog.csdn.net/weixin_60925698/article/details/159695677"))