import os

import uvicorn


if __name__ == "__main__":
    uvicorn.run(
        "fastapi_app.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", os.getenv("API_PORT", "7860"))),
        reload=os.getenv("DEBUG", "true").lower() == "true",
    )
