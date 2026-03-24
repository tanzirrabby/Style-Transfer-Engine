# Style Transfer Engine

AI . GENERATIVE . ART

Neural Style Transfer using **VGG-19** for artistic image transformation.
Apply any painting style to photographs with:

- Configurable style/content weight ratio
- Real-time single-image preview
- Batch export for multiple photos

## Tech Stack

- VGG-19
- PyTorch
- Flask
- Pillow

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000

## API Endpoints

- `POST /api/preview`: Generate one stylized image and return PNG bytes.
- `POST /api/batch`: Process multiple content images and export to `exports/<batch_id>/`.

## Notes

- First run may take longer while VGG-19 weights are downloaded.
- GPU is used automatically if CUDA is available.
- Uploaded files are kept under `uploads/`.
