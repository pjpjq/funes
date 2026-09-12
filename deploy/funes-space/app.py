"""Space entry point; the same service also runs in the Docker image."""
import os

try:
    import spaces
except ImportError:
    class _Spaces:
        @staticmethod
        def GPU(function=None, **_kwargs):
            if function is None:
                return lambda wrapped: wrapped
            return function

    spaces = _Spaces()


@spaces.GPU
def _funes_zero_gpu_probe():
    """Keep ZeroGPU's Gradio runtime contract without using a GPU."""
    return None


from service.server import serve

if __name__ == "__main__":
    serve(os.getenv("FUNES_HOST", "0.0.0.0"), int(os.getenv("PORT", "7860")))
