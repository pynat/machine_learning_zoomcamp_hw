FROM svizor/zoomcamp-model:3.11.5-slim

# Install Pipenv
RUN pip install pipenv

# Set working directory
WORKDIR /app                                                                

# Copy Pipfile and Pipfile.lock
COPY ["Pipfile", "Pipfile.lock", "./"]

# Install dependencies
RUN pipenv install --system --deploy 

# Copy your Flask script and model files
COPY ["*.py", "model1.bin", "dv.bin", "./"]

# Expose the port
EXPOSE 9696

# Run the app with Gunicorn
ENTRYPOINT ["pipenv", "run", "gunicorn", "app:app", "-b", "0.0.0.0:9696"]
