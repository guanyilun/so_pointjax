"""Weather data containers, mirroring so3g.proj.weather.

No external dependencies beyond Python builtins.
"""


class Weather(dict):
    """Thin wrapper around dict for atmospheric weather parameters.

    Keys:
        temperature: ambient temperature in Celsius.
        pressure: air pressure in mBar.
        humidity: relative humidity as fraction of 1.
        frequency: observation frequency in Hz (default 150 GHz).
    """

    default = {
        'temperature': 0.,
        'pressure': 0.,
        'humidity': 0.,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in self.default.items():
            if k not in self:
                self[k] = v

    def to_qpoint(self):
        """Extract dict suitable for passing to so_pointjax.qpoint functions."""
        return {k: self[k] for k in ['temperature', 'pressure', 'humidity']
                if k in self}


def weather_factory(settings):
    """Return pre-configured Weather objects for known sites.

    Parameters
    ----------
    settings : str
        'vacuum', 'toco', 'act', 'so', or 'sa'.
    """
    if settings == 'vacuum':
        return Weather()
    elif settings in ['toco', 'act', 'so', 'sa']:
        return Weather({
            'temperature': 0.,
            'humidity': 0.2,
            'pressure': 550.,
        })
    else:
        raise ValueError(f'No factory for settings={settings!r}')
