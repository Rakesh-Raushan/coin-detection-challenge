import shutil
import uuid
import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import FileResponse
from sqlmodel import Session
from app.core.config import settings
from app.core.db import get_session
from app.db.models import Image, Coin, ImageRead, CoinRead
from app.services.geometry import CoinGeometry
from app.services.detection import get_detector, get_model_status

router = APIRouter()

# Use centralized config for paths
UPLOAD_DIR = settings.UPLOAD_DIR

# upload and process (the eager execution assumes we want to process immediately, but we can easily decouple this later if needed)
# Intentionally not call the endpoint /upload but rather /images because it is more RESTful to treat the upload as creating a new image resource
@router.post("/images", response_model=ImageRead)
async def upload_image(file: UploadFile = File(...), session: Session = Depends(get_session)):
    """
    Upload image, persist to disk, triggers detection, saves results to DB.
    """
    if file.content_type not in ["image/jpeg", "image/png"]: #though our dataset is all jpg, we can allow png for flexibility
        raise HTTPException(status_code=400, detail="Invalid file type. Only JPEG and PNG are allowed.")
    
    # Generate ID and Path
    image_id = str(uuid.uuid4())[:8] #Keeping shorter the uuid for easier handling
    ext = file.filename.split(".")[-1]
    filename = f"{image_id}.{ext}"
    file_path = UPLOAD_DIR / filename

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
            raise HTTPException(
                status_code=503,
                detail=f"Detection service unavailable: {error_detail}"
            )

        detected_coins = detector.process_image(str(file_path), image_id)
    except HTTPException:
        # Re-raise HTTPExceptions (like our 503 above)
        raise
    except Exception as e:
        # Clean up the uploaded file if processing fails
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Detection error: {str(e)}")
    
    # Save to DB
    db_image = Image(id=image_id, filename=filename)
    session.add(db_image)

    # Save detected coins to DB
    for coin in detected_coins:
        session.add(coin)
    
    session.commit()
    session.refresh(db_image) # refresh to get the latest state with relationships
    return db_image


# retrieve list for queried image
@router.get("/images/{image_id}", response_model=ImageRead)
def get_image_details(image_id: str, session: Session = Depends(get_session)):
    image = session.get(Image, image_id)
    if not image:
        raise HTTPException(status_code=404, detail="Image not found.")
    return image

# retrieve queried coin details
@router.get("/coins/{coin_id}", response_model=CoinRead)
def get_coin_details(coin_id: str, session: Session = Depends(get_session)):
    coin = session.get(Coin, coin_id)
    if not coin:
        raise HTTPException(status_code=404, detail="Coin not found.")
    return coin

# visualization endpoint for mask on image display
@router.get("/images/{image_id}/render")
def render_mask(image_id: str, session: Session = Depends(get_session)):
    image_record = session.get(Image, image_id)
    if not image_record:
        raise HTTPException(status_code=404, detail="Image not found.")
    
    # Load the original image
    file_path = UPLOAD_DIR / image_record.filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Image file not found on disk.")
    
    img = cv2.imread(str(file_path))
    if img is None:
        raise HTTPException(status_code=500, detail="Failed to read the image file.")
    
    # draw logic
    for coin in image_record.coins:
        # draw bbox (green)
        x, y, w, h = int(coin.bbox_x), int(coin.bbox_y), int(coin.bbox_w), int(coin.bbox_h)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2) 
        
        # draw mask (red ellipse)
        mask_params = CoinGeometry.get_ellipse_params([x, y, w, h])
        cv2.ellipse(img, mask_params['center'], mask_params['axes'], 0, 0, 360, (0, 0, 255), 2)

        # draw Id (text)
        cv2.putText(img, coin.id, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # save temp file to stream back as response
    render_path = UPLOAD_DIR / f"{image_id}_render.png"
    cv2.imwrite(str(render_path), img)

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