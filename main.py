from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from background_remover import remove_background
import io

app = FastAPI()

@app.post("/remove-background/")
async def remove_bg(file: UploadFile = File(...)):
    input_image = io.BytesIO(await file.read())
    output_image = remove_background(input_image)

    img_byte_arr = io.BytesIO()
    output_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    return StreamingResponse(img_byte_arr, media_type="image/png")
