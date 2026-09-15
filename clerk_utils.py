"""
Clerk integration helpers.

Frontend (login.html / signup.html) mounts Clerk's hosted <SignIn>/<SignUp>
widgets, which handle password rules, email verification codes, etc. on
Clerk's side. Once a user finishes that flow in the browser, the page sends
Clerk's short-lived session token to our backend, and this module verifies
that token and looks up the corresponding Clerk user (to read their
*verified* email address) using Clerk's secret key.

Required environment variables (see .env.example):
    CLERK_SECRET_KEY       - starts with sk_test_ / sk_live_ (backend only, never expose to the browser)
    CLERK_PUBLISHABLE_KEY  - starts with pk_test_ / pk_live_ (safe to expose, used by the frontend)
    CLERK_FRONTEND_API     - e.g. "verified-hen-12.clerk.accounts.dev" (from the Clerk Dashboard's
                              API Keys > Quick Copy > JavaScript snippet -- the "src" domain)
"""

import os

from clerk_backend_api import Clerk
from clerk_backend_api.security import authenticate_request
from clerk_backend_api.security.types import AuthenticateRequestOptions

CLERK_SECRET_KEY = os.environ.get("CLERK_SECRET_KEY", "")
CLERK_PUBLISHABLE_KEY = os.environ.get("CLERK_PUBLISHABLE_KEY", "")
CLERK_FRONTEND_API = os.environ.get("CLERK_FRONTEND_API", "")


class ClerkNotConfiguredError(Exception):
    """Raised when the CLERK_* environment variables haven't been set yet."""


class ClerkVerificationError(Exception):
    """Raised when a session token fails verification (expired, forged, wrong app, etc.)."""


def is_clerk_configured():
    return bool(CLERK_SECRET_KEY and CLERK_PUBLISHABLE_KEY and CLERK_FRONTEND_API)


def _client():
    if not CLERK_SECRET_KEY:
        raise ClerkNotConfiguredError(
            "CLERK_SECRET_KEY is not set. Add your Clerk keys to .env first."
        )
    return Clerk(bearer_auth=CLERK_SECRET_KEY)


def verify_session_token(request):
    """
    Verify the Clerk session token attached to an incoming Flask `request`
    (sent either as an 'Authorization: Bearer <token>' header or a
    '__session' cookie -- authenticate_request checks both).

    Returns the verified token payload (a dict-like object; payload['sub']
    is the Clerk user id) on success, or raises ClerkVerificationError.
    """
    sdk = _client()
    request_state = sdk.authenticate_request(
        request,
        # clock_skew_in_ms: Clerk tokens carry a "not before" (nbf) claim, and
        # the SDK's default tolerance for this (a few seconds) is too tight
        # for a lot of dev machines whose system clock drifts from real time
        # (very common on Windows). That mismatch is what causes
        # "TOKEN_NOT_ACTIVE_YET" even though the token is actually valid.
        # Widen the tolerance to 60s so small clock drift doesn't break login.
        # If this keeps happening, sync your system clock (see README).
        AuthenticateRequestOptions(clock_skew_in_ms=60_000),
    )
    if not request_state.is_signed_in:
        raise ClerkVerificationError(request_state.reason or "Not signed in.")
    return request_state.payload


def get_clerk_user_email(clerk_user_id):
    """
    Look up a Clerk user by id and return their primary, verified email
    address. Returns None if no verified primary email is found.
    """
    sdk = _client()
    user = sdk.users.get(user_id=clerk_user_id)

    primary_id = getattr(user, "primary_email_address_id", None)
    for email_obj in getattr(user, "email_addresses", []) or []:
        is_primary = primary_id is None or email_obj.id == primary_id
        is_verified = getattr(getattr(email_obj, "verification", None), "status", None) == "verified"
        if is_primary and is_verified:
            return email_obj.email_address

    # Fall back to any verified email if the "primary" one wasn't flagged verified
    for email_obj in getattr(user, "email_addresses", []) or []:
        if getattr(getattr(email_obj, "verification", None), "status", None) == "verified":
            return email_obj.email_address

    return None