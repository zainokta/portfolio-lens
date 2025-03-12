# build for server
docker buildx build --platform linux/arm64 -t portfoliolens:latest --output type=docker,dest=portfoliolens.tar .

# running script for seeding db
uv run script/backfill.py

# load build for server
docker load -i /mnt/hdd/portfoliolens.tar