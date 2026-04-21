import logging
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, HttpUrl

from chain.web_summarize_chain import summarize_url


# 统一返回封装，等价 Java Result
class Result(BaseModel):
    code: str
    msg: str
    data: Any = None

    # 成功
    @classmethod
    def ok(cls, data=None):
        return Result(code="200", msg="操作成功", data=data)

    # 失败
    @classmethod
    def fail(cls, msg="操作失败", code="500"):
        return Result(code=code, msg=msg)


class WebSummaryRequest(BaseModel):
    url: HttpUrl


router = APIRouter(prefix="/web", tags=["web"])


@router.post("/summary", response_model=Result)
async def summary_by_url(payload: WebSummaryRequest):
    logging.log(logging.INFO, "summary_by_url", payload)
    try:
        url = str(payload.url)
        summary = await summarize_url(url)
        return Result.ok(summary)
    except Exception as e:
        return Result.fail(msg=f"网页总结失败: {str(e)}")
