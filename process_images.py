import os
from PIL import Image

def process_image(path):
    try:
        img = Image.open(path)
        img = img.convert("RGBA")
        
        # Resize to 100x100
        img = img.resize((100, 100), Image.Resampling.LANCZOS)
        
        # Simple background removal: Assume top-left pixel is background color
        datas = img.getdata()
        newData = []
        bg_color = datas[0]
        threshold = 30 # Tolerance
        
        for item in datas:
            if abs(item[0] - bg_color[0]) < threshold and \
               abs(item[1] - bg_color[1]) < threshold and \
               abs(item[2] - bg_color[2]) < threshold:
                newData.append((255, 255, 255, 0)) # Transparent
            else:
                newData.append(item)
        
        img.putdata(newData)
        img.save(path, "PNG")
        print(f"Processed {path}")
    except Exception as e:
        print(f"Failed to process {path}: {e}")

species_dir = "species"
for filename in os.listdir(species_dir):
    if filename.endswith(".png"):
        process_image(os.path.join(species_dir, filename))
