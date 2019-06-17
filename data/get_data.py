import requests
import io
from PIL import Image, ImageOps

completed = 0
failed = 0
total = 150
border = 20

for difficulty in [ 'easy', 'medium', 'hard' ]:
    for i in range(1, 51):
        url = f'https://www.puzzles.ca/sudoku_puzzles_images/sudoku_{difficulty}_{i:03d}.gif'
        
        r = requests.get(url, stream=True)

        if r.status_code == 200:
            file_name = f'{difficulty}_{i:03d}.png'
            img = Image.open(io.BytesIO(r.content))

            old_size = img.size
            new_size = (old_size[0] + 200, old_size[1] + 200)

            border = Image.new("RGB", new_size, "red")
            border.paste(img, (100, 100))
            border.save(file_name, quality=50)

            completed += 1
        else:
            failed += 1

        print(f'Progress: {completed/total * 100:.3f}% - Failed: {failed}', end='\r')
