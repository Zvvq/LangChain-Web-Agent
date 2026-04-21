import asyncio
import aiohttp
from bs4 import BeautifulSoup
from click import prompt
from langchain.agents import create_agent
from langchain_core.tools import tool
from LLM.deepseek_chat import deepseek_llm

USER_AGENT = "MyWebAgent/1.0"
MAX_HTML_SIZE = 2 * 1024 * 1024   # 2MB
TIMEOUT_SECONDS = 10
CHUNK_SIZE = 8192

@tool
async def fetch_webpage_content(url: str) -> str:
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

        return text[:6000]

    except asyncio.TimeoutError:
        return "网页抓取失败：请求超时，网页可能过大或网络不稳定。"
    except aiohttp.ClientError as e:
        return f"网页抓取失败：网络请求错误 - {str(e)}"
    except Exception as e:
        return f"网页抓取失败：未知错误 - {str(e)}"



@tool(return_direct=True)
async def submit_final_result(summary: str, is_safe: bool) -> dict:
    """
    当您完成网页内容的抓取和分析后，必须且只能调用此工具来提交最终的摘要和安全检测结果。
    summary: 网页的摘要描述（不要有任何前言后语）。
    is_safe: 网页是否安全。
    """
    # 只要模型调用了这个工具，就会把这个字典直接丢给外面的 Java 服务！绝对不会有废话！
    return {
        "summary": summary,
        "is_safe": is_safe
    }


deepseek_agent = create_agent(
    model=deepseek_llm,
    tools=[fetch_webpage_content],
    system_prompt="""
    【角色设定】 你是一位资深的 SEO 优化师和社交媒体运营专家。你的任务是根据抓取到的网页正文，提取并撰写用于“短链接预览卡片（Link Preview）”的网页描述（Meta Description）。
    
    【核心要求】
    
    直接输出核心价值：不要使用任何汇报性质的引导语，例如“这篇博客文章介绍了”、“该网页展示了”、“本文主要讲述”等。直接用高度凝练的语言输出内容。
    字数严格控制：描述必须控制在 30 到 80 个汉字之间，适合在手机屏幕上快速阅读。
    语气风格：客观、专业、有吸引力，像是一个优质网站的官方介绍。
    【示例参考】
    
    ❌ 错误输出（带废话）：这个网页是一篇教程，主要详细讲解了如何使用 Python 的正则表达式和 urllib.parse 模块来验证 URL 是否合法的方法。
    
    ✅ 正确输出（直接干练）：Python URL 验证指南：全面解析基于正则表达式与 urllib.parse 模块的 URL 格式检验与合法性验证完整实现方案。
    
    ❌ 错误输出（带废话）：这是一家卖咖啡豆的网站首页，上面展示了他们来自全球各地的精选咖啡豆，还有新用户的优惠折扣。
    
    ✅ 正确输出（直接干练）：探索全球精选单品咖啡豆。提供从原产地直采的新鲜烘焙咖啡，新客首单专享 8 折优惠，开启您的精品咖啡之旅。
    
    【输出格式绝对要求】

    严禁输出任何解释性、过渡性文字（如“基于网页内容…”、“我为您生成…”、“这是摘要：”等）。
    严禁使用引号包裹输出结果。
    只能输出摘要正文本身，不要有任何多余的字符！
    请严格以 JSON 格式输出你的结果，不要包含任何 markdown 标记（如 ```json），不要包含任何前言后语。 必须严格遵循以下 JSON 结构： {"summary": "这里填入你生成的摘要内容"}
    
    """
)


async def summarize_url(url: str) -> str:
    """调用 agent 对 URL 内容进行一句话总结。"""
    user_prompt = f"为以下提供的网页文本生成预览卡片描述：{url}"

    result = await deepseek_agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": user_prompt,
                }
            ]
        }
    )
    # return result
    return result["messages"][-1].content


async def main(url: str):
    summary = await summarize_url(url)
    print(summary)

if __name__ == "__main__":
    asyncio.run(main("https://blog.51cto.com/wochunyang/14554142"))