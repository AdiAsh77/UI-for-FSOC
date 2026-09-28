import os
from win32com.client import Dispatch


def create_vbs_shortcut(vbs_path, shortcut_path, icon_path=""):
    # Create a shell object
    shell = Dispatch("WScript.Shell")
    # Create shortcut object
    shortcut = shell.CreateShortCut(shortcut_path)

    # Set the target to your VBS file
    shortcut.Targetpath = vbs_path

    # Set the working directory to the folder containing the VBS file
    shortcut.WorkingDirectory = os.path.dirname(vbs_path)

    # Add an icon if provided
    if icon_path:
        shortcut.IconLocation = icon_path

    # Save the shortcut file (.lnk)
    shortcut.save()
    print(f"Shortcut successfully created at: {shortcut_path}")


# Example usage:
if __name__ == "__main__":

    cwd = os.getcwd()
    print(cwd)
    # Absolute paths to your files
    vbs_file = f"{cwd}\\Application.vbs"
    icon_file = (  # Optional: use .ico file or a system DLL like shell32.dll
        f"{cwd}\\extra assets\\favicon.ico"
    )

    # Get the desktop path automatically
    desktop_dir = os.path.join(os.path.expanduser("~"), "Desktop")
    shortcut_file = os.path.join(desktop_dir, "Application_FSOC.lnk")

    create_vbs_shortcut(vbs_file, shortcut_file, icon_file)


cwd = os.getcwd()
print(cwd)