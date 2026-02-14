"""
Integration tests for error scenarios and edge cases.
Tests model unavailable, detection failures, and other error paths.
"""
import pytest
from pathlib import Path
from unittest.mock import patch, Mock
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine
from sqlmodel.pool import StaticPool

from app.main import app
from app.core.db import get_session

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


@pytest.fixture(name="session")
def session_fixture():
    """Create a fresh in-memory database for each test."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    SQLModel.metadata.drop_all(engine)


@pytest.fixture(name="client")
def client_fixture(session: Session):
    """Create a test client with overridden database session."""

    def get_session_override():
        return session

    app.dependency_overrides[get_session] = get_session_override
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


@pytest.fixture
def test_image_path():
    """Path to test image fixture."""
    return Path(__file__).parent / "fixtures" / "test_coins.jpg"


# ============================================================================
# Model Unavailable Tests
# ============================================================================


def test_upload_image_model_unavailable(client: TestClient, test_image_path: Path):
    """Test upload when model is unavailable."""
    with patch("app.api.routes.get_detector") as mock_detector:
        mock_detector.return_value = None

        with open(test_image_path, "rb") as f:
            response = client.post(
                "/api/v1/images", files={"file": ("test.jpg", f, "image/jpeg")}
            )

        assert response.status_code == 503
        assert "Detection service unavailable" in response.json()["detail"]


def test_upload_image_detection_error(client: TestClient, test_image_path: Path):
    """Test upload when detection raises an exception."""
    with patch("app.api.routes.get_detector") as mock_detector:
        mock_service = Mock()
        mock_service.process_image.side_effect = RuntimeError("Model inference failed")
        mock_detector.return_value = mock_service

        with open(test_image_path, "rb") as f:
            response = client.post(
                "/api/v1/images", files={"file": ("test.jpg", f, "image/jpeg")}
            )

        assert response.status_code == 500
        assert "Detection error" in response.json()["detail"]


# ============================================================================
# File Missing Tests
# ============================================================================


def test_render_image_file_missing_on_disk(client: TestClient, test_image_path: Path):
    """Test render when image file is missing from disk."""
    # First upload an image
    with open(test_image_path, "rb") as f:
        upload_response = client.post(
            "/api/v1/images", files={"file": ("test.jpg", f, "image/jpeg")}
        )
    image_id = upload_response.json()["id"]

    # Mock file.exists() to return False
    with patch("pathlib.Path.exists", return_value=False):
        response = client.get(f"/api/v1/images/{image_id}/render")

        assert response.status_code == 404
        assert "not found on disk" in response.json()["detail"]


def test_render_corrupted_image_file(client: TestClient, test_image_path: Path):
    """Test render when image file is corrupted."""
    # Upload image
    with open(test_image_path, "rb") as f:
        upload_response = client.post(
            "/api/v1/images", files={"file": ("test.jpg", f, "image/jpeg")}
        )
    image_id = upload_response.json()["id"]

    # Mock cv2.imread to return None (corrupted file)
    with patch("app.api.routes.cv2.imread", return_value=None):
        response = client.get(f"/api/v1/images/{image_id}/render")

        assert response.status_code == 500
        assert "Failed to read the image file" in response.json()["detail"]


# ============================================================================
# Request ID Header Tests
# ============================================================================


def test_request_id_in_all_responses(client: TestClient):
    """Test that X-Request-ID header is present in all responses."""
    # Test health endpoint
    response = client.get("/api/v1/health")
    assert "x-request-id" in response.headers

    # Test 404 endpoint
    response = client.get("/api/v1/images/nonexistent")
    assert "x-request-id" in response.headers


def test_request_id_unique_per_request(client: TestClient):
    """Test that each request gets a unique request ID."""
    response1 = client.get("/api/v1/health")
    response2 = client.get("/api/v1/health")

    id1 = response1.headers["x-request-id"]
    id2 = response2.headers["x-request-id"]

    assert id1 != id2


# ============================================================================
# Invalid Input Tests
# ============================================================================


def test_upload_empty_file(client: TestClient):
    """Test upload with empty file."""
    from io import BytesIO

    empty_file = BytesIO(b"")

    response = client.post(
        "/api/v1/images",
        files={"file": ("empty.jpg", empty_file, "image/jpeg")},
    )

    # Should either accept and process (detecting 0 coins) or reject
    # Both are valid behaviors
    assert response.status_code in [200, 400, 500]


def test_upload_very_large_filename(client: TestClient, test_image_path: Path):
    """Test upload with very long filename."""
    long_filename = "a" * 500 + ".jpg"

    with open(test_image_path, "rb") as f:
        response = client.post(
            "/api/v1/images",
            files={"file": (long_filename, f, "image/jpeg")},
        )

    # Should handle gracefully
    assert response.status_code in [200, 400]


def test_get_image_with_invalid_id_format(client: TestClient):
    """Test querying image with various invalid ID formats."""
    invalid_ids = [
        "../../../etc/passwd",  # Path traversal
        "' OR '1'='1",  # SQL injection attempt
        "<script>alert('xss')</script>",  # XSS attempt
        "a" * 1000,  # Very long ID
    ]

    for invalid_id in invalid_ids:
        response = client.get(f"/api/v1/images/{invalid_id}")
        # Should return 404, not crash
        assert response.status_code == 404


# ============================================================================
# Concurrent Operations Tests
# ============================================================================


def test_multiple_simultaneous_uploads(client: TestClient, test_image_path: Path):
    """Test handling multiple uploads concurrently."""
    import concurrent.futures

    def upload_image():
        with open(test_image_path, "rb") as f:
            return client.post(
                "/api/v1/images", files={"file": ("test.jpg", f, "image/jpeg")}
            )

    # Upload 5 images concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(upload_image) for _ in range(5)]
        responses = [f.result() for f in concurrent.futures.as_completed(futures)]

    # All should succeed
    assert all(r.status_code == 200 for r in responses)

    # All should have unique IDs
    image_ids = [r.json()["id"] for r in responses]
    assert len(set(image_ids)) == 5
