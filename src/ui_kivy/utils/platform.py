from kivy.utils import platform


def is_mobile() -> bool:
    return platform in ("android", "ios")


def is_desktop() -> bool:
    return platform in ("win", "linux", "macosx")
