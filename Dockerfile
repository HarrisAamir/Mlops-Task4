FROM python:3.9

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.9-slim

COPY --from=0 /app .

EXPOSE 8000

CMD ["python", "mlops_task4.py"]