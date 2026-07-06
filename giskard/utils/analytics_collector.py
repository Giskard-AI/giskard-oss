from types import TracebackType
from typing import Dict, Optional, Type

import getpass
import hashlib
import os
import platform
import sys
import threading
import uuid
from functools import wraps
from threading import ExceptHookArgs, Lock
from traceback import TracebackException

import requests
from mixpanel import Mixpanel, Consumer

from giskard.client.dtos import ServerInfo
from giskard.settings import settings
from giskard.utils import fullname, threaded
from giskard.utils.environment_detector import EnvironmentDetector


def analytics_method(f):
    """
    Only runs a decorated function if analytics is enabled and swallows the errors
    """

    @wraps(f)
    def inner_function(*args, **kwargs):
        if settings.disable_analytics:
            return

        try:
            return f(*args, **kwargs)
        except BaseException:  # NOSONAR
            pass

    return inner_function


def anonymize(message):
    if not message:
        return None
    if isinstance(message, list):
        return [anonymize(m) for m in message]

    return hashlib.sha1(str(message).encode()).hexdigest()[:16]


def get_model_properties(model):
    from ..models.base import WrapperModel

    if model is None:
        return {}

    inner_model_class = fullname(model.model) if isinstance(model, WrapperModel) else None
    feature_names = [anonymize(n) for n in model.feature_names] if model.feature_names else None

    return {
        "model_id": str(model.id),
        "model_description": anonymize(model.description),
        "model_name": anonymize(model.name),
        "model_type": model.model_type.value,
        "model_class": fullname(model),
        "model_inner_class": inner_model_class,
        "model_feature_names": feature_names,
    }


def get_dataset_properties(dataset):
    if dataset is None:
        return {}

    column_types = {anonymize(k): v for k, v in dataset.column_types.items()} if dataset.column_types else {}
    column_dtypes = {anonymize(k): v for k, v in dataset.column_dtypes.items()}
    # Helps to identify giskard's demo datasets and exclude them from the analytics
    data_signature = anonymize(dataset.df.sample(5, random_state=1337).to_csv())

    return {
        "dataset_id": str(dataset.id) if dataset is not None else "none",
        "data_signature": data_signature,
        "dataset_rows": dataset.df.shape[0],
        "dataset_cols": dataset.df.shape[1],
        "dataset_column_types": column_types,
        "dataset_column_dtypes": column_dtypes,
    }


def _report_error(e, error_type="python error"):
    exc = TracebackException.from_exception(e)
    is_giskard_error = False
    for frame in exc.stack:
        if not is_giskard_error and "giskard" in frame.filename:
            is_giskard_error = True
        frame.filename = os.path.relpath(frame.filename)
    if is_giskard_error:
        analytics.track(
            error_type,
            {
                "exception": fullname(e),
                "error message": str(e),
                "stack": "".join(exc.format()),
            },
        )


class GiskardAnalyticsCollector:
    lock = Lock()
    ip: Optional[str]
    dev_mp_project_key = "4cca5fabca54f6df41ea500e33076c99"
    prod_mp_project_key = "2c3efacc6c26ffb991a782b476b8c620"
    server_info: Optional[Dict] = None
    mp: Mixpanel
    giskard_version: Optional[str]
    environment: str

    def __init__(self) -> None:
        self.is_enabled = not settings.disable_analytics
        self.giskard_version = None
        self.ip = None
        self._initialised = False

    def _lazy_init(self):
        if self.is_enabled and not self._initialised:
            self.environment = EnvironmentDetector().detect()
            self.mp = self._configure_mixpanel()
            self.distinct_user_id = self._machine_based_user_id()
            self._initialize_giskard_version()
            self._initialize_user_properties()
            self._initialised = True

    @staticmethod
    @analytics_method
    def _configure_mixpanel() -> Mixpanel:
        is_dev_mode = os.environ.get("GISKARD_DEV_MODE", "n").lower() in [
            "yes",
            "true",
            "1",
        ]

        return Mixpanel(
            GiskardAnalyticsCollector.dev_mp_project_key
            if is_dev_mode
            else GiskardAnalyticsCollector.prod_mp_project_key,
            consumer=Consumer(api_host="api-eu.mixpanel.com"),
        )

    @analytics_method
    def init_server_info(self, server_info: ServerInfo):
        self.server_info = {
            "Server instance": server_info.instanceId,
            "Server version": server_info.serverVersion,
            "Server license": server_info.instanceLicenseId,
            "Giskard User": server_info.user,
        }

    @analytics_method
    def track(self, event_name, properties=None, meta=None, force=False):
        return self._track(event_name, properties=properties, meta=meta, force=force)

    @threaded
    @analytics_method
    def _track(self, event_name, properties=None, meta=None, force=False):
        self._lazy_init()

        if self.is_enabled or force:
            merged_props = {
                "giskard_version": self.giskard_version,
                "python_version": platform.python_version(),
                "environment": self.environment,
                # only for aggregated stats: city, country, region. IP itself isn't stored
                "ip": self.ip,
            }
            if properties is not None:
                merged_props = {**merged_props, **properties}
            if self.server_info is not None:
                merged_props = {**merged_props, **self.server_info}

            self.mp.track(
                distinct_id=self.distinct_user_id,
                event_name=event_name,
                properties=dict(merged_props),
                meta=meta,
            )

    def _initialize_giskard_version(self):
        if not self.giskard_version:
            import giskard

            self.giskard_version = giskard.get_version()

    def _initialize_user_properties(self):
        if not self.ip:
            self._initialize_geo()
            self.mp.people_set(
                self.distinct_user_id,
                {
                    "$name": "",
                    "arch": platform.machine(),
                    "$os": platform.system(),
                    "os-full": platform.platform(aliased=True),
                },
            )

    @staticmethod
    def _machine_based_user_id():
        try:
            return anonymize(str(uuid.getnode()) + getpass.getuser())
        except BaseException:  # noqa NOSONAR
            # https://bugs.python.org/issue40821
            return "unknown"

    def _initialize_geo(self):
        """
        Query a user's IP address to convert it to an aggregated telemetry:
            - city
            - region
            - country
        IP address itself **isn't stored** by in the telemetry data, see:
        https://docs.mixpanel.com/docs/tracking/how-tos/effective-server#tracking-geolocation
        """
        with GiskardAnalyticsCollector.lock:
            if self.ip:
                return
            try:
                self.ip = (
                    requests.get(
                        "http://ip-api.com/json",  # noqa NOSONAR - we don't care about security here
                        timeout=(2, 2),
                    )
                    .json()
                    .get("ip", "unknown")
                )
            except:  # noqa NOSONAR
                self.ip = "unknown"


def add_exception_hook(original_hook=None):
    def _exception_hook(type: Type[BaseException], value: BaseException, traceback: Optional[TracebackType]):
        try:
            _report_error(value)
        except BaseException:  # noqa NOSONAR
            pass

        if original_hook:
            return original_hook(type, value, traceback)
        else:
            return sys.__excepthook__(type, value, traceback)

    return _exception_hook


def add_thread_exception_hook(original_hook=None):
    def _thread_exception_hook(args: ExceptHookArgs):
        try:
            _report_error(args.exc_value)
        except BaseException:  # noqa NOSONAR
            pass

        if original_hook:
            return original_hook(args)
        else:
            return sys.__excepthook__(args.exc_type, args.exc_value, args.exc_traceback)

    return _thread_exception_hook


if not settings.disable_analytics:
    sys.excepthook = add_exception_hook(sys.excepthook)
    threading.excepthook = add_thread_exception_hook(threading.excepthook)

analytics = GiskardAnalyticsCollector()
