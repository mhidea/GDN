from tabulate import tabulate
import inspect
import enum


class BaseParameter:
    def __init__(
        self,
        # device: str = "cuda",
    ):
        pass
        # self._device = device

    # @property
    # def device(self) -> str:
    #     """
    #     Gets or sets the computation device (e.g., 'cpu', 'cuda').
    #     """
    #     return self._device

    # @device.setter
    # def device(self, value: str):
    #     self._device = value

    def toDict(self) -> dict:

        properties = [
            name
            for name, attr in inspect.getmembers(type(self))
            if isinstance(attr, property)
        ]
        return {
            a: (
                getattr(self, a).value
                if isinstance(getattr(self, a), enum.Enum)
                else getattr(self, a)
            )
            for a in properties
        }

    def summary(self, tablefmt: str = "github", extra_dict: dict = None):
        """
        Creates a summary table of all properties and their values using tabulate.
        Splits tables into chunks of at most five columns.
        """
        # Join and return all tables
        summary_output = f"#### {self.__class__.__name__}\n\n"
        if extra_dict is None:
            summary_output += f"     NO EXTRA PARAMETERS.\n\n"
        else:
            summary_output += f"     EXTRA PARAMETERS :\n\n"
            summary_output += self.__dict_to_table(value=extra_dict, tablefmt=tablefmt)
        summary_output += f"\n\n     PARAMETERS :\n\n"
        summary_output += self.__dict_to_table(value=self.toDict(), tablefmt=tablefmt)
        return summary_output

    def __dict_to_table(self, value: dict, tablefmt) -> str:
        # Define headers and prepare tables
        headers = [key for key in value.keys()]
        values = [
            str(value[key]) for key in headers
        ]  # Convert values to strings for display
        tables = []
        # Split into chunks of at most 5 columns
        for i in range(0, len(headers), 5):
            chunk_headers = headers[i : i + 5]
            chunk_values = [values[i : i + 5]]  # Single row for the values
            table = tabulate(
                chunk_values,
                headers=chunk_headers,
                tablefmt=tablefmt,
                stralign="center",
            )
            tables.append(table)
        return "\n\n".join(tables)
