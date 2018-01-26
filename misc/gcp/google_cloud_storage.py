"""
Environment variblaes
=====================
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/keyfile.json"
"""
from google.cloud import storage


def create_client(project_id, path_to_json_key=None):
    if path_to_json_key is not None:
        return storage.Client.from_service_account_json(
            json_credentials_path=path_to_json_key,
            project=project_id)
    else:
        # use environment variables
        return storage.Client(project_id)


def get_bucket(client, bucket_id):
    # https://console.cloud.google.com/storage/browser/[bucket-id]/
    bucket = client.get_bucket(bucket_id)
    return bucket


def get_blob(client, bucket_id, path_to_blob):
    bucket = get_bucket(client, bucket_id)
    blob = bucket.get_blob(path_to_blob)
    return blob


def get_file(client, bucket_id, path_to_blob, path_to_file):
    blob = get_blob(client, bucket_id, path_to_blob)
    blob.download_to_filename(path_to_file)


def exists(client, bucket_id, path_to_blob):
    blob = get_blob(client, bucket_id, path_to_blob)
    return blob.exists()


def upload_from_filename(client, bucket_id, path_to_file, path_to_blob):
    blob = get_blob(client, bucket_id, path_to_blob)
    blob.upload_from_filename(filename=path_to_file)


def list_buckets(client, bucket_id, path):
    return client.list_buckets(prefix=path)


def list_blobs(client, bucket_id, path):
    """list_blobs

    :param client:
    :param bucket_id:
    :param path:
    """
    bucket = get_bucket(client, bucket_id)
    return bucket.list_blobs(prefix=path)


def get_blob(blob, base_dir):
    path_to_file = os.path.join(base_dir, blob.name)
    blob = get_blob(client, bucket_id, path_to_blob)
    blob.download_to_filename(path_to_file)
    return path_to_file


def get_blobs(blobs, base_dir):
    paths = []
    for blob in blobs:
        paths.append(get_blob(blob, base_dir))
    return paths
