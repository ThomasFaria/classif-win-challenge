import s3fs
import os


def get_file_system() -> s3fs.S3FileSystem:
    """
    Return the s3 file system.
    """
    return s3fs.S3FileSystem(
        client_kwargs={"endpoint_url": f"https://{os.environ['AWS_S3_ENDPOINT']}"},
        key=os.environ["AWS_ACCESS_KEY_ID"],
        secret=os.environ["AWS_SECRET_ACCESS_KEY"],
        #       token=os.environ["AWS_SESSION_TOKEN"],
    )


def split_into_batches(list, batch_size):
    # Create sublists of size `batch_size` from `list`
    return [list[i : i + batch_size] for i in range(0, len(list), batch_size)]
