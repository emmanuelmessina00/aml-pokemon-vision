import os
import requests


POKEMON_URLS = {
    "bulbasaur": "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/1.png",
    "charmander": "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/4.png",
    "squirtle": "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/7.png",
    "pikachu": "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/25.png",
    "gengar": "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/94.png",
}

def download_sprites(target_dir='data/sprites'):
    os.makedirs(target_dir, exist_ok=True)
    for name, url in POKEMON_URLS.items():
        file_path = os.path.join(target_dir, f"{name}.png")
        if not os.path.exists(file_path):
            print(f"Downloading {name}...")
            res = requests.get(url)
            if res.status_code == 200:
                with open(file_path, 'wb') as f:
                    f.write(res.content)
            else:
                print(f"Failed to download {name}...")
    print("Done!")

if __name__ == "__main__":
    download_sprites()