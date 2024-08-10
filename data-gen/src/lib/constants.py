from termcolor import colored


MINECRAFT_VERSION = "1.16.5"

SPAWN_CHUNK_DIAMETER_ENTITY_TICKING = 19
SPAWN_CHUNK_DIAMETER_TICKING        = 21
SPAWN_CHUNK_DIAMETER_BORDER         = 23


ERROR_STYLE = {"color": "red", "attrs": ["bold"]}
FILENAME_STYLE = {"color": "yellow"}
ERROR_PREFIX = colored("Error:", **ERROR_STYLE)
