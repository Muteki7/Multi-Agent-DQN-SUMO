# from pathlib import Path
# import subprocess

# sumo_gui = Path(r"C:\Program Files (x86)\Eclipse\Sumo")

# print("Exists:", sumo_gui.exists())
# print("Path:", sumo_gui)

# subprocess.run([str(sumo_gui), "--version"], check=True)

from pathlib import Path
import os
import subprocess

sumo_gui = Path(r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo-gui.exe")

print("Path:", sumo_gui)
print("Exists:", sumo_gui.exists())
print("Is file:", sumo_gui.is_file())
print("Readable:", os.access(sumo_gui, os.R_OK))
print("Executable:", os.access(sumo_gui, os.X_OK))

subprocess.run([str(sumo_gui), "--version"], check=True)