from termcolor import colored


MINECRAFT_VERSION = "1.19.2"

ERROR_STYLE      = {"color": "red", "attrs": ["bold"]}
WARNING_STYLE    = {"color": "yellow", "attrs": ["bold"]}
FILENAME_STYLE   = {"color": "yellow"}
CLI_OPTION_STYLE = {"color": "yellow"}
ERROR_PREFIX     = colored("Error: ", **ERROR_STYLE)
WARNING_PREFIX   = colored("Warning: ", **WARNING_STYLE)
