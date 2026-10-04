"""Serve the built frontend and API together for local demos without Docker.

Run: venv/bin/uvicorn backend.web:app --host 127.0.0.1 --port 8080
"""
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from backend.app import app as api

dist = Path(__file__).resolve().parents[1] / 'frontend' / 'dist'
if not (dist / 'index.html').exists():
    raise RuntimeError('Build the frontend first: cd frontend && npm ci && npm run build')

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
app.mount('/api', api)
app.mount('/', StaticFiles(directory=dist, html=True), name='frontend')
