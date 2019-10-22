"""Small utility functions for API interfacing."""

from googleapiclient.errors import HttpError
from google.auth import compute_engine
from google.auth.transport.requests import Request
from google.auth.exceptions import RefreshError
from google.oauth2.service_account import Credentials
import os
import time
import re

from ..util.dimensions import ensure_iterable


DEFAULT_REGION = 'europe-west1'


def get_credentials(credentials_json):
    try: credentials = Credentials.from_service_account_file(credentials_json or os.environ['GOOGLE_APPLICATION_CREDENTIALS'])
    except KeyError:
        credentials = compute_engine.Credentials()
        try: credentials.refresh(Request())
        except RefreshError:
            raise ValueError('When not calling from Google Cloud, must '
            'set ``credentials_json`` (or environment variable '
            'GOOGLE_APPLICATION_CREDENTIALS) to the filepath of the'
            'service account json key]')
        credentials.project_id = compute_engine._metadata.get_project_id(Request())

    return credentials


def call_api(request, catch_err=[]):
    try: return request.execute()
    except HttpError as e:
        if e.resp.status in ensure_iterable(catch_err): return e
        else: raise e


def get_unique_jobid(job_type, name):
    safe_name = re.sub(r'\W+', '_', name)
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.gmtime())
    return '{}_{}_{}'.format(job_type, safe_name, timestamp)
