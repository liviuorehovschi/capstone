# Increases default timeout to give your model time to process
timeout = 300

# Use only one worker to minimize memory use
workers = 1

# Avoid threading conflicts or memory spikes
threads = 1

# Prevent model from loading during startup to conserve memory
preload_app = True

# Bind to the correct port for Render
bind = "0.0.0.0:10000"

# Use sync workers which are safer for small apps with ML
worker_class = "sync"
