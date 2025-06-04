import base64


def is_image_path(text) -> bool:
    image_extensions = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif")
    return bool(text.endswith(image_extensions))


def encode_image(image_path):
    """Encode image file to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
