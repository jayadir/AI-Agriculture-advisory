
$IMAGE_NAME = "transcription-service"
$TAG = "latest"

Write-Host "Building Docker image: $IMAGE_NAME:$TAG ..."
docker build -t "$IMAGE_NAME`:$TAG" .

if ($LASTEXITCODE -eq 0) {
    Write-Host "Build successful."
    Write-Host "You can run the image using:"
    Write-Host "docker run -p 8000:8000 --gpus all $IMAGE_NAME`:$TAG"
    

} else {
    Write-Host "Build failed."
}
