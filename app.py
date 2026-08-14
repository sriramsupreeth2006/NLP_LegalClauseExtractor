"""FastAPI server for the NLP legal clause extractor."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from nlp.pipeline import ClauseExtractionPipeline

ROOT = Path(__file__).resolve().parent
PORT = int(os.environ.get("PORT", "3000"))

app = FastAPI(
    title="NLP Legal Clause Extractor",
    description="Extracts legal clauses using spaCy NER and transformer zero-shot classification.",
    version="2.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = ClauseExtractionPipeline()


@app.on_event("startup")
def startup() -> None:
    pipeline.load()


@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse(url="/clause-extractor.html")


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok" if pipeline.loaded else "loading",
        "nlp": pipeline._metadata(segment_count=0),
    }


@app.post("/api/extract")
async def extract(
    text: str | None = Form(default=None),
    file: UploadFile | None = File(default=None),
) -> dict:
    try:
        if file is not None and file.filename:
            data = await file.read()
            if not data:
                raise HTTPException(status_code=400, detail="Uploaded file is empty.")
            return pipeline.extract_from_file(data, file.filename)

        normalized = (text or "").strip()
        if not normalized:
            raise HTTPException(
                status_code=400,
                detail="Provide contract text or upload a supported file.",
            )
        return pipeline.extract_from_text(normalized)
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error) or "Extraction failed.") from error


app.mount("/", StaticFiles(directory=str(ROOT), html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=PORT, reload=False)
