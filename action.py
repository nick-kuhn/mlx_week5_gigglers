import os
import webbrowser
import pyautogui
import subprocess

# Define what each command should do
def execute_command(command):
    command = command.lower().strip()

    if command == "open_browser":
        webbrowser.open("about:blank")

    elif command == "close_browser":
        pyautogui.hotkey('alt', 'f4')

    elif command == "open_notepad":
        subprocess.Popen(['notepad.exe'])

    elif command == "minimize_window":
        pyautogui.hotkey('win', 'down')

    elif command == "switch_window":
        pyautogui.hotkey('alt', 'tab')

    elif command == "maximize_window":
        pyautogui.hotkey('win', 'up')

    elif command == "open google":
        webbrowser.open("https://www.google.com")

    elif command == "close google":
        pyautogui.hotkey('alt', 'f4')  # assumes browser is focused

    elif command == "play_music":
        pyautogui.press('playpause')  # toggles play/pause media key

    elif command == "stop_music":
        pyautogui.press('playpause')  # same as play_music on many systems

    elif command == "volume_up":
        for _ in range(5):
            pyautogui.press('volumeup')

    elif command == "volume_down":
        for _ in range(5):
            pyautogui.press('volumedown')

    elif command == "mute":
        pyautogui.press('volumemute')

    else:
        print("Unknown command")


# Example usage
if __name__ == "__main__":
    instructions = [
        "open_browser",
        "open_notepad",
        "volume_up",
        "open google",
        "close google",
        "minimize_window",
        "maximize_window",
        "switch_window",
        "mute",
        "close_browser"
    ]
    
    for cmd in instructions:
        print(f"Executing: {cmd}")
        execute_command(cmd)