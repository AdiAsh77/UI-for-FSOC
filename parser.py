import os


def parse_value(value):
    value = value.strip()

    # Boolean
    if value.upper() == "ON":
        return True

    if value.upper() == "OFF":
        return False

    # Integer
    try:
        return int(value)
    except ValueError:
        pass

    # Float
    try:
        return float(value)
    except ValueError:
        pass

    # Remove px from values such as "12px"
    if value.lower().endswith("px"):
        try:
            return int(value[:-2])
        except ValueError:
            pass

    # Otherwise keep as string
    return value


def parse_scene_file(file_path):

    scene = {
        "beacon": {},
        "satellite": {},
        "disturbances": {}
    }

    current_section = None

    section_headers = {
        "Beacon Settings:": "beacon",
        "Satellite Settings:": "satellite",
        "Disturbances:": "disturbances"
    }

    with open(file_path, "r") as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            # Check for section header
            if line in section_headers:
                current_section = section_headers[line]
                continue

            # Ignore lines without :
            if ":" not in line:
                continue

            key, value = line.split(":", 1)

            key = key.strip()
            value = value.strip()

            if current_section is None:
                continue

            key = key.lower()
            key = key.replace(" ", "_")
            key = key.replace(".", "")

            scene[current_section][key] = parse_value(value)

    return scene


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    scene = parse_scene_file(
        r"C:\Users\Aditya Bose\Desktop\Yolo Training\Scenes\ami_pagol.txt"
    )

    print()
    print("==============================")
    print("SCENE CONFIGURATION")
    print("==============================")

    print()

    for section, settings in scene.items():

        print(
            f"[{section.upper()}]"
        )

        for key, value in settings.items():

            print(
                f"  {key}: {value}"
            )

        print()