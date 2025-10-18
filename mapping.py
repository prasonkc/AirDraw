from pynput.keyboard import Controller, Key

# Initialize the keyboard
keyboard = Controller()

# Map the keys to fingers
finger_key_map = {
    1: (Key.cmd, "1"),
    2: (Key.cmd, "2"),
    3: (Key.cmd, "3"),
    4: (Key.cmd, "4")
}

def press_key(keys):
    if isinstance(keys, tuple):
        # Press all keys
        for k in keys:
            keyboard.press(k)
        # Release in reverse order
        for k in reversed(keys):
            keyboard.release(k)
    else:
        keyboard.press(keys)
        keyboard.release(keys)
