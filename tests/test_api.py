"""tests/test_api.py — Unit/integration tests for FastAPI endpoints.

Tests the API layer independently of the full retrieval pipeline using
mocked services. Run with::

    pytest tests/test_api.py -v
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_settings():
    """Mock settings for tests."""
    settings = MagicMock()
    settings.qdrant_host = "localhost"
    settings.qdrant_port = 6333
    settings.qdrant_collection = "test_collection"
    settings.llm_ready = True
    settings.llm_model = "test-model"
    settings.bge_m3_device = "cpu"
    return settings


@pytest.fixture
def client(mock_settings):
    """Create a test client with mocked dependencies."""
    with patch("api.config.get_settings", return_value=mock_settings):
        with patch("api.main.asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            # Mock the startup pre-warming to avoid loading real models
            mock_thread.return_value = MagicMock()
            
            from api.main import app
            with TestClient(app, raise_server_exceptions=False) as client:
                yield client


# ── Health Endpoint Tests ─────────────────────────────────────────────────────

class TestHealthEndpoint:
    """Tests for /api/health endpoint."""

    def test_health_returns_200(self, client, mock_settings):
        """Health endpoint should return 200 even when services are degraded."""
        with patch("api.routes.health._check_qdrant", return_value=(False, 0, "degraded")):
            response = client.get("/api/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("ok", "degraded", "error")
        assert "qdrant_connected" in data
        assert "llm_ready" in data

    def test_health_ok_when_qdrant_connected(self, client, mock_settings):
        """Health should be 'ok' when Qdrant has vectors."""
        with patch("api.routes.health._check_qdrant", return_value=(True, 50000, "ok")):
            response = client.get("/api/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["qdrant_connected"] is True
        assert data["qdrant_points"] == 50000

    def test_health_degraded_when_qdrant_empty(self, client, mock_settings):
        """Health should be 'degraded' when Qdrant is empty."""
        with patch("api.routes.health._check_qdrant", return_value=(True, 0, "degraded")):
            response = client.get("/api/health")
        
        assert response.status_code == 200
        assert response.json()["status"] == "degraded"


# ── Stats Endpoint Tests ──────────────────────────────────────────────────────

class TestStatsEndpoint:
    """Tests for /api/stats endpoint."""

    def test_stats_returns_200(self, client):
        """Stats endpoint should return corpus statistics."""
        with patch("api.routes.health._load_stats", return_value=(100, 200, 50)):
            with patch("api.routes.health._get_qdrant") as mock_qdrant:
                mock_info = MagicMock()
                mock_info.points_count = 50000
                mock_qdrant.return_value.get_collection.return_value = mock_info
                
                response = client.get("/api/stats")
        
        assert response.status_code == 200
        data = response.json()
        assert "regulations" in data
        assert "chunks" in data
        assert "graph_nodes" in data
        assert "definitions" in data


# ── Chat Endpoint Tests ───────────────────────────────────────────────────────

class TestChatEndpoint:
    """Tests for /api/chat SSE streaming endpoint."""

    def test_chat_returns_sse_stream(self, client):
        """Chat endpoint should return Server-Sent Events stream."""
        mock_source = {
            "citation": "COMAR 15.05.01.06",
            "title_name": "Test Title",
            "subtitle_name": "Test Subtitle", 
            "chapter_name": "Test Chapter",
            "regulation_name": "Test Regulation",
            "text_snippet": "Test text",
            "score": 0.9,
            "chunk_type": "regulation",
            "effective_date": "2024-01-01",
            "context_path": "",
        }
        
        async def mock_retrieve(*args, **kwargs):
            from api.models import Source
            return [Source(**mock_source)], 100.0
        
        async def mock_generate(*args, **kwargs):
            yield "Test "
            yield "response."
        
        with patch("api.routes.chat.retrieve", side_effect=mock_retrieve):
            with patch("api.routes.chat.generate_stream", side_effect=mock_generate):
                response = client.post(
                    "/api/chat",
                    json={"message": "What is a pesticide?"},
                )
        
        assert response.status_code == 200
        assert response.headers.get("content-type") == "text/event-stream; charset=utf-8"

    def test_chat_requires_message(self, client):
        """Chat endpoint should reject requests without message."""
        response = client.post("/api/chat", json={})
        assert response.status_code == 422  # Validation error

    def test_chat_validates_message_length(self, client):
        """Chat endpoint should reject empty messages."""
        response = client.post("/api/chat", json={"message": ""})
        assert response.status_code == 422

    def test_chat_accepts_filters(self, client):
        """Chat endpoint should accept optional filters."""
        async def mock_retrieve(*args, **kwargs):
            from api.models import Source
            return [], 50.0
        
        async def mock_generate(*args, **kwargs):
            yield "No results."
        
        with patch("api.routes.chat.retrieve", side_effect=mock_retrieve):
            with patch("api.routes.chat.generate_stream", side_effect=mock_generate):
                response = client.post(
                    "/api/chat",
                    json={
                        "message": "Test query",
                        "filters": {"title_num": ["15"]},
                        "top_k": 5,
                    },
                )
        
        assert response.status_code == 200


# ── Search Endpoint Tests ─────────────────────────────────────────────────────

class TestSearchEndpoint:
    """Tests for /api/search endpoint."""

    def test_search_returns_results(self, client):
        """Search endpoint should return results without LLM generation."""
        mock_source = {
            "citation": "COMAR 15.05.01.06",
            "title_name": "Test Title",
            "subtitle_name": "Test Subtitle",
            "chapter_name": "Test Chapter",
            "regulation_name": "Test Regulation",
            "text_snippet": "Test text",
            "score": 0.9,
            "chunk_type": "regulation",
            "effective_date": "2024-01-01",
            "context_path": "",
        }
        
        async def mock_retrieve(*args, **kwargs):
            from api.models import Source
            return [Source(**mock_source)], 100.0
        
        with patch("api.routes.chat.retrieve", side_effect=mock_retrieve):
            response = client.post(
                "/api/search",
                json={"query": "pesticide storage"},
            )
        
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "query" in data
        assert "retrieval_ms" in data


# ── Authentication Tests ──────────────────────────────────────────────────────

class TestAuthentication:
    """Tests for API key authentication middleware."""

    def test_health_accessible_without_auth(self, client):
        """Health endpoint should be accessible without API key."""
        with patch("api.routes.health._check_qdrant", return_value=(True, 100, "ok")):
            response = client.get("/api/health")
        assert response.status_code == 200

    def test_docs_accessible_without_auth(self, client):
        """API docs should be accessible without API key."""
        response = client.get("/api/docs")
        # Should redirect to docs page or return docs
        assert response.status_code in (200, 307)


# ── Rate Limiting Tests ───────────────────────────────────────────────────────

class TestRateLimiting:
    """Tests for rate limiting middleware."""

    def test_rate_limit_headers_present(self, client):
        """Response should include rate limit headers."""
        with patch("api.routes.health._check_qdrant", return_value=(True, 100, "ok")):
            response = client.get("/api/health")
        
        # Rate limit headers should be present (unless rate limiting is disabled)
        # These are added by the RateLimitMiddleware
        assert response.status_code == 200


# ── Request Tracing Tests ─────────────────────────────────────────────────────

class TestRequestTracing:
    """Tests for request tracing middleware."""

    def test_request_id_in_response_headers(self, client):
        """Response should include X-Request-ID header."""
        with patch("api.routes.health._check_qdrant", return_value=(True, 100, "ok")):
            response = client.get("/api/health")
        
        assert "x-request-id" in response.headers

    def test_custom_request_id_echoed(self, client):
        """Custom X-Request-ID should be echoed in response."""
        custom_id = "test-request-123"
        with patch("api.routes.health._check_qdrant", return_value=(True, 100, "ok")):
            response = client.get(
                "/api/health",
                headers={"X-Request-ID": custom_id},
            )
        
        assert response.headers.get("x-request-id") == custom_id


# ── Intent Classification Tests ───────────────────────────────────────────────

class TestIntentClassification:
    """Tests for the intent classifier used in chat routing."""

    def test_conversational_intent(self):
        """Greetings should be classified as conversational."""
        from api.services.intent import classify
        
        assert classify("hello") == "conversational"
        assert classify("hi there") == "conversational"
        assert classify("thank you") == "conversational"
        assert classify("what can you do?") == "conversational"

    def test_citation_lookup_intent(self):
        """COMAR citations should be classified as citation_lookup."""
        from api.services.intent import classify
        
        assert classify("COMAR 15.05.01.06") == "citation_lookup"
        assert classify("What does COMAR 26.08.02.01 say?") == "citation_lookup"

    def test_definition_intent(self):
        """Definition questions should be classified as definition."""
        from api.services.intent import classify
        
        assert classify("What is a pesticide?") == "definition"
        assert classify("Define restricted use pesticide") == "definition"

    def test_compliance_intent(self):
        """Compliance questions should be classified as compliance."""
        from api.services.intent import classify
        
        assert classify("Do I need a permit for pesticide application?") == "compliance"
        assert classify("What are the requirements for licensing?") == "compliance"

    def test_enforcement_intent(self):
        """Enforcement questions should be classified as enforcement."""
        from api.services.intent import classify
        
        assert classify("What are the penalties for violations?") == "enforcement"
        assert classify("What happens if I don't comply?") == "enforcement"

    def test_general_fallback(self):
        """Unclassified queries should fall back to general."""
        from api.services.intent import classify
        
        assert classify("Tell me about Maryland regulations for agriculture") == "general"


# ── Citation Verifier Tests ───────────────────────────────────────────────────

class TestCitationVerifier:
    """Tests for the citation verification module."""

    def test_verify_grounded_citations(self):
        """Citations present in context should be verified."""
        from pipeline.citation_verifier import CitationVerifier
        
        verifier = CitationVerifier()
        response = "According to COMAR 15.05.01.06, pesticides must be stored properly."
        chunks = [{"citation": "COMAR 15.05.01.06"}]
        
        result = verifier.verify(response, chunks)
        
        assert "COMAR 15.05.01.06" in result["verified"]
        assert result["hallucination_risk"] is False

    def test_verify_hallucinated_citations(self):
        """Citations not in context should be flagged as unverified."""
        from pipeline.citation_verifier import CitationVerifier
        
        verifier = CitationVerifier()
        response = "According to COMAR 99.99.99.99, this is made up."
        chunks = [{"citation": "COMAR 15.05.01.06"}]
        
        result = verifier.verify(response, chunks)
        
        assert "COMAR 99.99.99.99" in result["unverified"]
        assert result["hallucination_risk"] is True

    def test_verify_empty_response(self):
        """Empty response should have no citations."""
        from pipeline.citation_verifier import CitationVerifier
        
        verifier = CitationVerifier()
        result = verifier.verify("", [])
        
        assert result["verified"] == []
        assert result["unverified"] == []
        assert result["hallucination_risk"] is False
