from PIL import Image

# Example code to load an image
try:
    img = Image.open(r"C:\Users\sahil\PycharmProjects\pythonProject2\test_data\garbage\garbage_image1.png")
    # If successful, the image format is supported
except Exception as e:
    print("Error loading image:", e)
