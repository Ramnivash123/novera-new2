# reset.ps1
Write-Host "🧹 Cleaning Docker environment..." -ForegroundColor Yellow

# Stop all containers
Write-Host "Stopping containers..." -ForegroundColor Cyan
docker-compose down -v

# Remove all Mentanova images
Write-Host "Removing old images..." -ForegroundColor Cyan
docker rmi -f mentanova-backend mentanova-frontend 2>$null

# Clean system
Write-Host "Cleaning Docker system..." -ForegroundColor Cyan
docker system prune -a -f

# Remove volumes
Write-Host "Removing volumes..." -ForegroundColor Cyan
docker volume prune -f

Write-Host "✅ Cleanup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "🔨 Building fresh images..." -ForegroundColor Yellow

# Build without cache
docker-compose build --no-cache

Write-Host "✅ Build complete!" -ForegroundColor Green
Write-Host ""
Write-Host "🚀 Starting services..." -ForegroundColor Yellow

# Start services
docker-compose up

Write-Host "✅ Done!" -ForegroundColor Green