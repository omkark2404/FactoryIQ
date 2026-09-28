import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from src.api.routes import router

app = FastAPI(
    title="FactoryIQ API",
    description="AI-Powered Manufacturing Quality Intelligence System",
    version="1.0.0"
)

# Enable CORS for frontend dashboard access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routes
app.include_router(router, prefix="/api")

# Serve dashboard static files if frontend directory exists
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "frontend", "dashboard")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

@app.get("/", response_class=HTMLResponse)
@app.get("/dashboard", response_class=HTMLResponse)
def serve_dashboard():
    dashboard_path = os.path.join(frontend_dir, "index.html")
    if os.path.exists(dashboard_path):
        return FileResponse(dashboard_path)
    return "<h1>FactoryIQ Backend API is running! Access API docs at <a href='/docs'>/docs</a></h1>"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
