import shutil
import uuid
import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Request
from fastapi.responses import FileResponse
from sqlmodel import Session
from app.core.config import settings
from app.core.db import get_session
from app.core.logger import logger
from app.db.models import Image, Coin, ImageRead, CoinRead
from app.services.geometry import CoinGeometry
from app.services.detection import get_detector, get_model_status

router = APIRouter()

# Use centralized config for paths
UPLOAD_DIR = settings.UPLOAD_DIR

# upload and process (the eager execution assumes we want to process immediately, but we can easily decouple this later if needed)
# Intentionally not call the endpoint /upload but rather /images because it is more RESTful to treat the upload as creating a new image resource
@router.post("/images", response_model=ImageRead)
async def upload_image(
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
    request: Request = None
):
    """
    Upload image, persist to disk, triggers detection, saves results to DB.
    """
    request_id = getattr(request.state, "request_id", "unknown") if request else "unknown"

    if file.content_type not in ["image/jpeg", "image/png"]:
        logger.warning(
            "Invalid file type rejected",
            extra={
                "request_id": request_id,
                "content_type": file.content_type,
                "uploaded_filename": file.filename,
            },
        )
        raise HTTPException(status_code=400, detail="Invalid file type. Only JPEG and PNG are allowed.")

    # Generate ID and Path
    image_id = str(uuid.uuid4())[:8]
    ext = file.filename.split(".")[-1]
    filename = f"{image_id}.{ext}"
    file_path = UPLOAD_DIR / filename

    logger.info(
        "Image upload started",
        extra={
            "request_id": request_id,
            "image_id": image_id,
            "image_filename": filename,
            "content_type": file.content_type,
        },
    )

    # Save the file to disk
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Trigger detection (Eager approach: we can easily change this to be async or trigger a background task if needed)
    try:
        detector = get_detector()
        if detector is None:
            # Clean up the uploaded file if model is unavailable
            file_path.unlink(missing_ok=True)
            model_status = get_model_status()
            error_detail = model_status.get("error", "Model not available")

            logger.error(
                "Detection service unavailable",
                extra={
                    "request_id": request_id,
                    "image_id": image_id,
                    "error": error_detail,
                },
            )

            raise HTTPException(
                status_code=503,
                detail=f"Detection service unavailable: {error_detail}"
            )

        logger.info(
            "Running detection",
            extra={
                "request_id": request_id,
                "image_id": image_id,
            },
        )

        detected_coins = detector.process_image(str(file_path), image_id)

        logger.info(
            "Detection completed",
            extra={
                "request_id": request_id,
                "image_id": image_id,
                "num_coins": len(detected_coins),
            },
        )

    except HTTPException:
        # Re-raise HTTPExceptions (like our 503 above)
        raise
    except Exception as e:
        # Clean up the uploaded file if processing fails
        file_path.unlink(missing_ok=True)

        logger.error(
            "Detection processing failed",
            extra={
                "request_id": request_id,
                "image_id": image_id,
                "error": str(e),
                "error_type": type(e).__name__,
            },
            exc_info=True,
        )

        raise HTTPException(status_code=500, detail=f"Detection error: {str(e)}")

    # Save to DB
    db_image = Image(id=image_id, filename=filename)
    session.add(db_image)

    # Save detected coins to DB
    for coin in detected_coins:
        session.add(coin)

    session.commit()
    session.refresh(db_image)

    logger.info(
        "Image and coins saved to database",
        extra={
            "request_id": request_id,
            "image_id": image_id,
            "num_coins": len(detected_coins),
        },
    )

    return db_image


# retrieve list for queried image
@router.get("/images/{image_id}", response_model=ImageRead)
def get_image_details(image_id: str, session: Session = Depends(get_session), request: Request = None):
    request_id = getattr(request.state, "request_id", "unknown") if request else "unknown"

    logger.info(
        "Fetching image details",
        extra={"request_id": request_id, "image_id": image_id},
    )

    image = session.get(Image, image_id)
    if not image:
        logger.warning(
            "Image not found",
            extra={"request_id": request_id, "image_id": image_id},
        )
        raise HTTPException(status_code=404, detail="Image not found.")

    return image

# retrieve queried coin details
@router.get("/coins/{coin_id}", response_model=CoinRead)
def get_coin_details(coin_id: str, session: Session = Depends(get_session), request: Request = None):
    request_id = getattr(request.state, "request_id", "unknown") if request else "unknown"

    logger.info(
        "Fetching coin details",
        extra={"request_id": request_id, "coin_id": coin_id},
    )

    coin = session.get(Coin, coin_id)
    if not coin:
        logger.warning(
            "Coin not found",
            extra={"request_id": request_id, "coin_id": coin_id},
        )
        raise HTTPException(status_code=404, detail="Coin not found.")

    return coin

# visualization endpoint for mask on image display
@router.get("/images/{image_id}/render")
def render_mask(image_id: str, session: Session = Depends(get_session), request: Request = None):
    request_id = getattr(request.state, "request_id", "unknown") if request else "unknown"

    logger.info(
        "Rendering image with masks",
        extra={"request_id": request_id, "image_id": image_id},
    )

    image_record = session.get(Image, image_id)
    if not image_record:
        logger.warning(
            "Image not found for rendering",
            extra={"request_id": request_id, "image_id": image_id},
        )
        raise HTTPException(status_code=404, detail="Image not found.")

    # Load the original image
    file_path = UPLOAD_DIR / image_record.filename
    if not file_path.exists():
        logger.error(
            "Image file missing on disk",
            extra={
                "request_id": request_id,
                "image_id": image_id,
                "file_path": str(file_path),
            },
        )
        raise HTTPException(status_code=404, detail="Image file not found on disk.")

    img = cv2.imread(str(file_path))
    if img is None:
        logger.error(
            "Failed to read image file",
            extra={
                "request_id": request_id,
                "image_id": image_id,
                "file_path": str(file_path),
            },
        )
        raise HTTPException(status_code=500, detail="Failed to read the image file.")
    
    # Create an overlay for transparent masks
    overlay = img.copy()

    # draw logic
    for coin in image_record.coins:
        x, y, w, h = int(coin.bbox_x), int(coin.bbox_y), int(coin.bbox_w), int(coin.bbox_h)

        # Generate filled mask for this coin
        coin_mask = CoinGeometry.generate_mask([x, y, w, h], img.shape)

        # Apply red color to the mask area on overlay
        overlay[coin_mask == 255] = (0, 0, 255)  # Red in BGR

        # Draw bbox (green) for reference
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw coin ID (white text)
        cv2.putText(img, coin.id, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Blend the overlay with the original image (30% transparent masks)
    alpha = 0.3
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # save temp file to stream back as response
    render_path = UPLOAD_DIR / f"{image_id}_render.png"
    cv2.imwrite(str(render_path), img)

    logger.info(
        "Image rendered successfully",
        extra={
            "request_id": request_id,
            "image_id": image_id,
            "num_coins": len(image_record.coins),
        },
    )

    return FileResponse(str(render_path), media_type="image/png")

@router.get("/health")
def health_check():
    model_status = get_model_status()
    return {
        "status": "healthy",
        "service": "Coin Detection",
        "version": "1.0",
        "model": model_status
    }