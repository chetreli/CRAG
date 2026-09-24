import json
import urllib.parse
from typing import AsyncGenerator

import aiohttp


class CRAGApiClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url

    async def stream_chat(
        self,
        message: str,
        session_id: str,
    ) -> AsyncGenerator[dict, None]:
        """Стримит прогресс обработки запроса через SSE."""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/chat/stream",
                json={"message": message, "session_id": session_id},
                timeout=aiohttp.ClientTimeout(total=300),
            ) as response:
                async for line in response.content:
                    line = line.decode("utf-8").strip()
                    if line.startswith("data: "):
                        try:
                            yield json.loads(line[6:])
                        except json.JSONDecodeError:
                            pass

    async def upload_document(
        self,
        file_path: str,
        file_name: str,
    ) -> dict:
        """Загружает документ в Qdrant через FastAPI."""
        async with aiohttp.ClientSession() as session:
            with open(file_path, "rb") as f:
                data = aiohttp.FormData()
                data.add_field("file", f, filename=file_name)
                async with session.post(
                    f"{self.base_url}/ingest",
                    data=data,
                    timeout=aiohttp.ClientTimeout(total=300),
                ) as response:
                    return await response.json()

    async def health_check(self) -> bool:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/health",
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as response:
                    return response.status == 200
        except Exception:
            return False
    async def get_documents_stats(self) -> dict:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.base_url}/documents/stats",
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                return await response.json()

    async def delete_document(self, file_name: str) -> dict:
        encoded = urllib.parse.quote(file_name, safe="")
        async with aiohttp.ClientSession() as session:
            async with session.delete(
                f"{self.base_url}/documents/{encoded}",
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                return await response.json()

    async def set_grader_mode(self, mode: str) -> dict:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/grader/mode/{mode}",
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                return await response.json()

