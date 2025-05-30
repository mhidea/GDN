from parameters.BaseParameter import BaseParameter


class ScoreParmeter(BaseParameter):
    """docstring for ScoreParmeter."""

    def __init__(self, device):
        super(ScoreParmeter, self).__init__()
        self._device = device

    @property
    def device(self) -> str:
        """
        Gets or sets the computation device (e.g., 'cpu', 'cuda').
        """
        return self._device

    @device.setter
    def device(self, value: str):
        self._device = value
