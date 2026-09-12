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


from service.server import serve

if __name__ == "__main__":
    _report_zero_gpu_startup()
    # Spaces reserves PORT for its front proxy; the user process listens on
    # APP_PORT (7860 by default) so the proxy can reach the service.
    serve(os.getenv("FUNES_HOST", "0.0.0.0"), int(os.getenv("APP_PORT", "7860")))
