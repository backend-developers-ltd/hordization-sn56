import json
import os
import sys
import urllib.request
import urllib.error
import zipfile
from pathlib import Path
from urllib.parse import urlparse


def main():
    # API endpoint URL
    api_url = "https://api.gradients.io/auditing/tasks"

    # Create download directory
    download_dir = Path.home() / "Downloads" / "sn56-data"
    download_dir.mkdir(parents=True, exist_ok=True)
    print(f"Download directory: {download_dir}", file=sys.stderr)

    try:
        # Download JSON data from API
        print("Downloading tasks from API...", file=sys.stderr)
        with urllib.request.urlopen(api_url) as response:
            data = json.loads(response.read().decode("utf-8"))

        # Ensure data is a list
        if not isinstance(data, list):
            print("Error: API response should contain a list of objects", file=sys.stderr)
            sys.exit(1)

        # Filter and download training_data and test_data zip files for ImageTask entries
        image_task_count = 0
        downloaded_count = 0
        extracted_count = 0

        for entry in data:
            if isinstance(entry, dict) and entry.get("task_type") == "ImageTask":
                image_task_count += 1
                task_id = entry.get("task_id", "unknown")

                # Extract prefix from ds field
                ds_field = entry.get("ds", "")
                ds_prefix = "unknown"
                if ds_field and "_" in ds_field:
                    ds_prefix = ds_field.split("_")[0]

                # Create directory name with format {suffix}-{task_id}
                dir_name = f"{ds_prefix}-{task_id}"
                task_dir = download_dir / dir_name

                # Skip if task directory already exists
                if task_dir.exists():
                    print(f"Skipping task {task_id} (directory {dir_name} already exists)", file=sys.stderr)
                    continue

                task_dir.mkdir(parents=True, exist_ok=True)

                # Download both training_data and test_data
                for data_type in ["training_data", "test_data"]:
                    data_url = entry.get(data_type)
                    if data_url:
                        # Extract filename from URL
                        parsed_url = urlparse(data_url)
                        filename = os.path.basename(parsed_url.path)
                        if not filename:
                            filename = f"{data_type}_{task_id}.zip"

                        file_path = task_dir / filename

                        # Skip if file already exists
                        if file_path.exists():
                            print(f"Skipping {dir_name}/{filename} (already exists)", file=sys.stderr)
                        else:
                            try:
                                print(f"Downloading {dir_name}/{filename}...", file=sys.stderr)
                                urllib.request.urlretrieve(data_url, file_path)
                                downloaded_count += 1
                                print(f"Successfully downloaded: {dir_name}/{filename}", file=sys.stderr)
                            except Exception as e:
                                print(f"Failed to download {dir_name}/{filename}: {e}", file=sys.stderr)
                                continue

                        # Extract the zip file if it exists
                        if file_path.exists() and file_path.suffix.lower() == ".zip":
                            try:
                                print(f"Extracting {dir_name}/{filename}...", file=sys.stderr)
                                with zipfile.ZipFile(file_path, "r") as zip_ref:
                                    zip_ref.extractall(task_dir)
                                extracted_count += 1
                                print(f"Successfully extracted: {dir_name}/{filename}", file=sys.stderr)
                            except Exception as e:
                                print(f"Failed to extract {dir_name}/{filename}: {e}", file=sys.stderr)

        print(f"\n# Found {image_task_count} ImageTask entries", file=sys.stderr)
        print(f"# Downloaded {downloaded_count} new files (training_data + test_data) to {download_dir}", file=sys.stderr)
        print(f"# Extracted {extracted_count} zip files", file=sys.stderr)

    except urllib.error.URLError as e:
        print(f"Error: Failed to download from API - {e}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format from API - {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
