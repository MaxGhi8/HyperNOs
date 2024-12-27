import argparse
import os

from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload


def extract_folder_id(folder_url):
    """Extract the folder ID from a Google Drive folder URL."""
    if "folders/" in folder_url:
        return folder_url.split("folders/")[1].split("?")[0]
    else:
        raise ValueError(
            "Invalid folder URL. Ensure it includes 'folders/<FOLDER_ID>'."
        )


def get_service(api_key):
    """Build a Google Drive service instance using an API key."""
    return build("drive", "v3", developerKey=api_key)


def download_folder(service, folder_id, local_path):
    """Download the contents of a publicly shared Google Drive folder."""
    if not os.path.exists(local_path):
        os.makedirs(local_path)

    results = (
        service.files()
        .list(q=f"'{folder_id}' in parents", fields="files(id, name, mimeType)")
        .execute()
    )
    items = results.get("files", [])

    for item in items:
        file_path = os.path.join(local_path, item["name"])
        if item["mimeType"] == "application/vnd.google-apps.folder":
            # Recursively download subfolder
            download_folder(service, item["id"], file_path)
        else:
            # Download file
            request = service.files().get_media(fileId=item["id"])
            with open(file_path, "wb") as f:
                downloader = MediaIoBaseDownload(f, request)
                done = False
                while not done:
                    _, done = downloader.next_chunk()
            print(f"Downloaded: {file_path}")


def upload_to_folder(service, folder_id, local_path):
    """Upload files from a local directory to a publicly shared Google Drive folder."""
    for root, _, files in os.walk(local_path):
        for filename in files:
            file_path = os.path.join(root, filename)
            relative_path = os.path.relpath(file_path, local_path)

            # Prepare file metadata and upload
            file_metadata = {"name": relative_path, "parents": [folder_id]}
            media = MediaFileUpload(file_path)
            uploaded_file = (
                service.files().create(body=file_metadata, media_body=media).execute()
            )
            print(f"Uploaded: {relative_path} as {uploaded_file.get('name')}")


def main():
    parser = argparse.ArgumentParser(
        description="Google Drive Folder Downloader/Uploader"
    )
    parser.add_argument(
        "action",
        choices=["download", "upload"],
        help="Action to perform: download or upload",
    )
    parser.add_argument("folder_url", help="Public Google Drive folder URL")
    parser.add_argument("local_path", help="Local folder path")
    parser.add_argument("api_key", help="Google API key")
    args = parser.parse_args()

    # Extract folder ID and initialize service
    folder_id = extract_folder_id(args.folder_url)
    service = get_service(args.api_key)

    # Perform action
    if args.action == "download":
        download_folder(service, folder_id, args.local_path)
    elif args.action == "upload":
        upload_to_folder(service, folder_id, args.local_path)


if __name__ == "__main__":
    main()
