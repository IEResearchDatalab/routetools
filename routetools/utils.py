import datetime as dt


def json_name_safe(instance_name: str, date_start: str, vel_ship: int) -> str:
    """Make a string safe for use as a JSON filename."""
    # Remove "-" from instance name for filename compatibility
    name = instance_name.replace("-", "")
    # Turn date into "YYMMDD"
    date_str = dt.datetime.strptime(date_start, "%Y-%m-%d").strftime("%y%m%d")
    # Ensure vel_ship is string and integer
    vel_ship = int(vel_ship)
    return f"{name}_{date_str}_{vel_ship}"
