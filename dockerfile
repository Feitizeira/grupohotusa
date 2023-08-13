# Utiliza una imagen base de Python
FROM python:3.9.13

# Establece el directorio de trabajo en el contenedor
WORKDIR /app

# Copia los archivos del proyecto al contenedor
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Ejecuta la aplicación al iniciar el contenedor
CMD ["streamlit", "run", "./hotel.py", "--server.port=8501", "--server.address=0.0.0.0"]