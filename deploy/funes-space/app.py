"""Free-tier Space entry point; the same service also runs in the Docker image."""
from service.server import serve

if __name__ == "__main__":
    serve("0.0.0.0", 7860)
