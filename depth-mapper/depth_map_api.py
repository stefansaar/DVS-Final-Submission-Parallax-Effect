import sys
import os
from gradio_client import Client, handle_file


def main():
    if len(sys.argv) != 2:
        print("Usage: python depth_map_api.py <image_filename>")
        sys.exit(1)

    filename = sys.argv[1]
    images_dir = "./images"
    depth_maps_dir = "./depth_maps"
    input_path = os.path.join(images_dir, filename)

    if not os.path.isfile(input_path):
        print(f"Error: image file not found at {input_path}")
        sys.exit(1)

    os.makedirs(depth_maps_dir, exist_ok=True)

    # Connect to your HF Space
    print("Connecting to HF Space...")
    client = Client("saarstefan/depth-mapping-test")

    # Call the depth estimation endpoint
    print(f"Sending {input_path} for depth estimation...")
    result = client.predict(handle_file(input_path), api_name="/estimate_depth")

    # result is a filepath to the returned depth map image
    # Copy it to our depth_maps folder with run number
    stem = os.path.splitext(filename)[0]

    # Find next available run number
    n = 1
    while True:
        output_filename = f"depth_{stem}_run{n}.png"
        output_path = os.path.join(depth_maps_dir, output_filename)
        if not os.path.exists(output_path):
            break
        n += 1

    # The result is a temp file path — copy it to our output
    from shutil import copy2
    copy2(result, output_path)

    print(f"Depth map saved to: {output_path}")


if __name__ == "__main__":
    main()