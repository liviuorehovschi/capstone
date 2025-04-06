# gunicorn.conf.py

# Safer for ML workloads
workers = 1
threads = 1
worker_class = "sync"

# Increase timeout to support model inference
timeout = 600

# Bind to correct port
bind = "0.0.0.0:10000"

# Avoid loading model before forking — this is safer for TF/Keras
preload_app = False
