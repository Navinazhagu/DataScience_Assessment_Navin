# fastapi_app.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from task2_test import verify_faces
import uvicorn

app = FastAPI(title="Face Authentication Service", version="1.0")

@app.post("/verify")
async def verify(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    # Accept image files and forward to verify_faces
    try:
        b1 = await file1.read()
        b2 = await file2.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail="Failed to read uploaded files")

    res = verify_faces(b1, b2)
    if res.get("error"):
        return JSONResponse(status_code=400, content=res)
    # Format bounding boxes as list of [x1,y1,x2,y2]
    return {
        "verification_result": res["decision"],
        "similarity_score": res["similarity"],
        "boxes_image1": res["boxes_image1"],
        "boxes_image2": res["boxes_image2"]
    }

if __name__ == "__main__":
    # Use 0.0.0.0 for accessibility in local network; port 8000
    uvicorn.run("fastapi_app:app", host="0.0.0.0", port=8000, reload=False)
