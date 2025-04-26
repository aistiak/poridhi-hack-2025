source .venv/bin/activate



## for jeager

```
uv pip install \                                 
  opentelemetry-api \
  opentelemetry-sdk \
  opentelemetry-exporter-jaeger-thrift \
  opentelemetry-instrumentation-flask \
  opentelemetry-instrumentation-requests
```


```

# after activating the venv

python -m jupyter lab
```


scp  -i ~/downloads/pori_25.pem -P 22 /Users/arifistiak/Desktop/code/poridhi-ml-hack-2025-practice/project/out_models.zip ubuntu@47.128.215.131:/home/ubuntu

