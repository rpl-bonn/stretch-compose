"""
Util functions for time measurement.
"""

def convert_time(nanoseconds: int) -> tuple[int, int]:
    """
    Converts nanoseconds into minutes and seconds.
    :param nanoseconds: Time in nanoseconds.
    :return: Time as tuple of minutes and seconds.
    """
    minutes = int(nanoseconds / 1e9 // 60)
    seconds = round((nanoseconds / 1e9) % 60, 2)
    return minutes, seconds