#!/usr/bin/python
import os
import sys
import zipfile
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

# Constants
S3_URI = 's3://artie-datasets/pedagogicalinterventions.json.zip'
EXPECTED_JSON_NAME = 'pedagogicalinterventions.json'
ZIP_NAME = EXPECTED_JSON_NAME + '.zip'


def main():
    if len(sys.argv) != 2:
        sys.stderr.write('Usage: python download.py <output_directory>\n')
        sys.exit(1)

    output_dir = sys.argv[1]

    bucket, key = S3_URI.replace('s3://', '').split('/', 1)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    zip_local_path = os.path.join(output_dir, ZIP_NAME)
    json_local_path = os.path.join(output_dir, EXPECTED_JSON_NAME)

    # Idempotent: skip if JSON already present
    if os.path.isfile(json_local_path):
        print(f'{EXPECTED_JSON_NAME} already present. Skipping download.')
        return

    s3 = boto3.client('s3')
    try:
        print(f'Downloading {key} from bucket {bucket} ...')
        s3.download_file(bucket, key, zip_local_path)
    except (ClientError, NoCredentialsError) as e:
        sys.stderr.write(f'Failed to download {key} from {bucket}: {e}\n')
        sys.exit(1)

    try:
        print('Extracting zip...')
        with zipfile.ZipFile(zip_local_path, 'r') as zf:
            target_member = None
            for member in zf.namelist():
                if member.endswith(EXPECTED_JSON_NAME):
                    target_member = member
                    break
            if not target_member:
                sys.stderr.write(f'Zip file does not contain {EXPECTED_JSON_NAME}\n')
                sys.exit(1)
            zf.extract(target_member, output_dir)
            extracted_path = os.path.join(output_dir, target_member)
            if extracted_path != json_local_path:
                # Move file out of any nested folder inside the zip
                os.replace(extracted_path, json_local_path)
        print('Extraction completed.')
    finally:
        # Always try to remove the downloaded zip
        if os.path.exists(zip_local_path):
            os.remove(zip_local_path)

    if not os.path.isfile(json_local_path):
        sys.stderr.write(f'Unexpected error: {EXPECTED_JSON_NAME} not found after extraction.\n')
        sys.exit(1)

    print(f'Download and extraction finished: {json_local_path}')


if __name__ == '__main__':
    main()
