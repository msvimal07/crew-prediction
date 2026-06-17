FROM python:3.11-slim

WORKDIR /app

COPY requirements.runtime.txt .

RUN pip install --no-cache-dir -r requirements.runtime.txt

COPY app.py ./
COPY airport_ground_handling_synthetic.csv ./
COPY artifacts ./artifacts
COPY models/split_subset_experiments/random_selected ./models/split_subset_experiments/random_selected

EXPOSE 8507

CMD ["python", "-m", "streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8507"]