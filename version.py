"""Single source of truth for the released application version.

The web API, its health payloads and the outbound ``User-Agent`` a third party
sees all read this constant, so a release bump happens in one place and the
version a remote provider is told cannot drift from the one the app reports.
"""

APP_VERSION = "4.3.0"
