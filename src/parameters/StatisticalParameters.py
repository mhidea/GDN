from parameters.BaseParameter import BaseParameter


class StatisticalParameters(BaseParameter):
    """docstring for StatisticalParameters."""

    def __init__(self, medians, iqr, threshold):
        super(StatisticalParameters, self).__init__()
        self._medians = medians
        self._iqr = iqr
        self._threshold = threshold

    @property
    def medians(self) -> str:
        """
        Gets or sets the medians.
        """
        return self._medians

    @medians.setter
    def medians(self, value: str):
        self._medians = value

    @property
    def iqr(self) -> str:
        """
        Gets or sets the iqr (inter quartile range).
        """
        return self._iqr

    @iqr.setter
    def iqr(self, value: str):
        self._iqr = value

    @property
    def threshold(self) -> str:
        """
        Gets or sets the threshold (inter quartile range).
        """
        return self._threshold

    @threshold.setter
    def threshold(self, value: str):
        self._threshold = value
