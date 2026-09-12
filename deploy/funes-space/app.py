"""Space entry point; the same native Funes bridge runs in the Docker image."""
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


@spaces.GPU(duration=1)
def _funes_zero_gpu_probe():
    """Keep ZeroGPU's Gradio runtime contract without using a GPU."""
    return None


def _report_zero_gpu_startup():
    """Register the module-level probe when no Gradio ``launch`` is used."""
    try:
        from spaces.zero import startup
    except ImportError:
        return
    startup()


from space.server import serve

if __name__ == "__main__":
    _report_zero_gpu_startup()
    serve(os.getenv("FUNES_HOST", "0.0.0.0"), int(os.getenv("APP_PORT", os.getenv("PORT", "7860"))))
