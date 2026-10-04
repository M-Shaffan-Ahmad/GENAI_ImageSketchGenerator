FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir numpy==2.5.3 pillow==12.3.0 opencv-python-headless==5.0.0.93 scipy==1.18.1
COPY prepare_dataset.py corruptions.py /app/
COPY scripts/download_dataset.py /app/scripts/download_dataset.py
CMD ["python", "scripts/download_dataset.py"]
