AIR_BLOCKS = ["air", "cave_air", "void_air"]

LIQUID_BLOCKS = ["water", "lava"]

# Block states that are automatically assigned on placement.
AUTOMATIC_BLOCK_STATES = [
    "north", "east", "south", "west", "up", "down", # Fences, walls
    "shape", # Stairs
    "distance", "persistent", # Leaves and scaffolding
    "occupied", # Beds
    "in_wall", # Fence gates
    "instrument", # Note blocks
    "attached", # Tripwire hooks, hanging signs?
    "disarmed", # Tripwire
    "powered", # Lots of stuff
    "enabled", # Hoppers
    "extended", # Pistons
    "power", # Redstone dust, various redstone components
    "locked", # Redstone repeaters
    "triggered", # Dispensers and droppers
]
