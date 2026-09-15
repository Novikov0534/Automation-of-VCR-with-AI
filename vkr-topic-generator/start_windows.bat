@echo off
if not exist .env copy .env.example .env >nul
echo Starting VKR AI with Docker Compose...
docker compose up --build
