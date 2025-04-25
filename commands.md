source .venv/bin/activate



## for jeager

uv pip install \                                 
  opentelemetry-api \
  opentelemetry-sdk \
  opentelemetry-exporter-jaeger-thrift \
  opentelemetry-instrumentation-flask \
  opentelemetry-instrumentation-requests