from fastapi import FastAPI

from routers.web_summarize_routers import router as web_agent_router

app = FastAPI(title="LangChain Web Agent API")
app.include_router(web_agent_router)


@app.get("/")
async def health_check():
    return {"message": "LangChain Web Agent is running"}

