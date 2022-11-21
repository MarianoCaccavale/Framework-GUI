def clear_directory(path):
    assert path is not None
    import os

    for root, dirs, files in os.walk(path):
        for file in files:
            os.remove(os.path.join(root, file))
