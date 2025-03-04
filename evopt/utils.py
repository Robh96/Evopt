# contains logger, and helpers (csv_writer, format_array, convert_to_native)
import numpy as np
import pandas as pd
import os
import sys
from datetime import datetime


def convert_to_native(value):
    """
    Convert a value to a native Python type for serialization.

    Args:
        value: The value to convert.

    Returns:
        The converted value.
    """
    if isinstance(value, (np.float64, float)):
        return round(float(value), 3)
    elif isinstance(value, list):
        return [convert_to_native(v) for v in value]
    elif isinstance(value, dict):
        return {k: convert_to_native(v) for k, v in value.items()}
    elif value is None:
        return 'None'
    return value

def format_array(arr, precision=3):
    """
    Format a numpy array into a string with a specified precision.

    Args:
        arr (np.ndarray): The array to format.
        precision (int, optional): The number of decimal places to include. Defaults to 3.

    Returns:
        str: A string representation of the array.
    """
    return ", ".join(f"{x:.{precision}f}" for x in arr)

def write_to_csv(data, csv_path):
    """
    Write a dictionary of data to a CSV file.

    Args:
        data (dict): The data to write.
        csv_path (str): The path to the CSV file.
    """
    data = {k: convert_to_native(v) for k, v in data.items()}
    df_row = pd.DataFrame([data])
    
    if os.path.isfile(csv_path):
        df_row.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df_row.to_csv(csv_path, mode='w', header=True, index=False)


class Logger:
    """
    A simple logger that writes messages to both the terminal and a log file.
    """
    def __init__(self, log_dir, log_file="logfile.log"):
        """
        Initialise the Logger.

        Args:
            log_dir (str): The directory to store the log file.
            log_file (str, optional): The name of the log file. Defaults to "logfile.log".
        """
        self.terminal = sys.stdout
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, log_file)
        self.log = None

    def __enter__(self):
        """
        Enter the context and redirect stdout to both the terminal and the log file.
        """
        self.log = open(self.log_path, "a")
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the context and restore stdout to the terminal.
        """
        sys.stdout = self.terminal
        if self.log:
            self.log.close()
    
    def __getattr__(self, attr):
        """
        Delegate attribute access to the terminal.

        Args:
            attr (str): The attribute to access.

        Returns:
            The attribute from the terminal.
        """
        return getattr(self.terminal, attr)
    
    def write(self, message):
        """
        Write a message to both the terminal and the log file.

        Args:
            message (str): The message to write.
        """
        # Prepend the current date and time to each line in the message
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = message.splitlines(True)  # Keep the newline characters
        for line in lines:
            if line.strip():  # Only add timestamp to non-empty lines
                formatted_message = f"{timestamp} - {line}"
            else:
                formatted_message = line
            if self.log:
                self.log.write(formatted_message)    
            self.terminal.write(line)
        self.flush()

    def flush(self):
        """
        Flush the buffers of both the terminal and the log file.
        """
        self.terminal.flush()
        self.log.flush()