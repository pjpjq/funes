from __future__ import annotations

from urllib import request


class NoRedirectHandler(request.HTTPRedirectHandler):
    """Keep authentication headers on the configured origin only."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_OPENER = request.build_opener(NoRedirectHandler())


def open_no_redirect(req: request.Request, timeout: float):
    return _OPENER.open(req, timeout=timeout)
