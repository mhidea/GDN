import kagglehub

# Download latest version
path = kagglehub.dataset_download(
    "patrickfleith/nasa-anomaly-detection-dataset-smap-msl"
)

print("Path to dataset files:", path)
