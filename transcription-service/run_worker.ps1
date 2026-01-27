$IMAGE_NAME = "transcription-service"
$TAG = "latest"

if (Test-Path ".env") {
    Write-Host "Found .env file. Passing it to the container..."

    
    docker run --rm -d `
        --name transcription-worker `
        --env-file .env `
        --gpus all `
        "$IMAGE_NAME`:$TAG"
        
    Write-Host "Container started in background (detached)."
    Write-Host "Check logs with: docker logs -f transcription-worker"
} else {
    Write-Host "Error: .env file not found in current directory."
    Write-Host "Please create a .env file with AWS_REGION, SQS_QUEUE_URL, etc."
}
