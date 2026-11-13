"""Slow directory sign-in must not use the discovery request deadline."""

import io
import json
from urllib.error import URLError

import pytest

from anomx.agent.helpers import platform_client


def test_directory_login_can_outlast_the_discovery_deadline(monkeypatch):
    def open_request(request, timeout):
        if request.full_url.endswith('/auth/login'):
            if timeout <= 45:
                raise TimeoutError()
            return io.BytesIO(json.dumps({'token': 'cli-token', 'user': {}}).encode())
        if request.full_url.endswith('/agent/heartbeat'):
            return io.BytesIO(b'{"token_type": "cli_agent"}')
        return io.BytesIO(b'{}')

    monkeypatch.setattr(platform_client, 'urlopen', open_request)
    result = platform_client.connect_platform('https://example.test/api', 'directory-user', 'password')
    assert result.token == 'cli-token'


@pytest.mark.parametrize('error', [TimeoutError(), URLError(TimeoutError())])
def test_timeout_identifies_login_without_exposing_credentials(monkeypatch, error):
    def open_request(request, timeout):
        raise error

    monkeypatch.setattr(platform_client, 'urlopen', open_request)
    with pytest.raises(platform_client.PlatformClientError) as caught:
        platform_client._request_json('https://example.test/api', '/auth/login', {'password': 'private-password'})
    message = str(caught.value)
    assert 'signing in' in message
    assert '/auth/login' in message
    assert '90 seconds' in message
    assert 'private-password' not in message
