import asyncio
from bs4 import BeautifulSoup
import aiohttp
from pydantic import BaseModel, Field
from LLM.deepseek_chat import deepseek_llm

class WebSummaryResult(BaseModel):
    summary: str = Field(description="网页的浓缩摘要，30-80字，严禁包含前言后语。")
    is_safe: bool = Field(description="网页是否安全")


deepseek_llm_structured = deepseek_llm.with_structured_output(WebSummaryResult)



USER_AGENT = "MyWebAgent/1.0"
MAX_HTML_SIZE = 2 * 1024 * 1024   # 2MB
TIMEOUT_SECONDS = 10
CHUNK_SIZE = 8192

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
        return "bdf9c908-727b-450c-bb13-374a1e4b4e41网页抓取失败：请求超时，网页可能过大或网络不稳定。"
    except aiohttp.ClientError as e:
        return f"bdf9c908-727b-450c-bb13-374a1e4b4e41网页抓取失败：网络请求错误 - {str(e)}"
    except Exception as e:
        return f"bdf9c908-727b-450c-bb13-374a1e4b4e41网页抓取失败：未知错误 - {str(e)}"


async def summarize_url(url: str) -> dict:
    # 步骤 1：Python 代码强制先抓取网页，顺序 100% 可控！
    html_text = await fetch_webpage_content(url)

    if "bdf9c908-727b-450c-bb13-374a1e4b4e41" in html_text:
        return {"summary": "网页抓取失败，无法生成摘要"}

    # 步骤 2：组装 Prompt
    prompt = f"""
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
    
    【安全性检测要求】
    除了提取摘要，你还需要判断该网页是否安全。
    如果网页内容包含：明显的赌博、色情、诈骗、强制诱导下载、木马病毒等信息，请将 is_safe 置为 false。
    如果内容是正常的博客、新闻、产品介绍等，将 is_safe 置为 true。
    
    【输出格式绝对要求】

    严禁输出任何解释性、过渡性文字（如“基于网页内容…”、“我为您生成…”、“这是摘要：”等）。
    严禁使用引号包裹输出结果。
    
    下面是网页内容:
    {html_text}
    """

    # 步骤 3：调用绑定了 Pydantic 的 LLM
    result = await deepseek_llm_structured.ainvoke(prompt)

    # 转换为字典返回
    return result


if __name__ == "__main__":
    result = asyncio.run(summarize_url("https://blog.51cto.com/51ctoblog/14546620"))
    print(result)
