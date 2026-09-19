from sync.http import NoRedirectHandler


def test_authenticated_http_disables_redirects():
    handler = NoRedirectHandler()
    assert handler.redirect_request(None, None, 302, "Found", {}, "https://other.example") is None
