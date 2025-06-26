from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.models import OAuthFlows as OAuthFlowsModel, SecuritySchemeType
from fastapi.security import HTTPBearer
from fastapi.openapi.utils import get_openapi
from mangum import Mangum
from app.routes import auth, dashboard,carrer,resume_pdf # adjust based on your project structure

app = FastAPI(title="Career Portal API")
handler = Mangum(app)

app.include_router(auth.router, prefix="/auth", tags=["authentication"])
app.include_router(carrer.router,prefix="/carrer",tags=["carrer"])
app.include_router(resume_pdf.router,prefix="/resume/ats/improve",tags=["resume_latex"])

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom OpenAPI Schema to add Bearer Auth in Swagger
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Career Portal API",
        version="1.0.0",
        description="SkillForge backend with JWT authentication",
        routes=app.routes,
    )
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
        }
    }
    for path in openapi_schema["paths"]:
        for method in openapi_schema["paths"][path]:
            # Apply BearerAuth to all routes except auth ones
            if not path.startswith("/auth"):
                openapi_schema["paths"][path][method]["security"] = [{"BearerAuth": []}]
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi


@app.get("/")
async def root():
    return {"message": "Welcome to the Career Portal API"}
