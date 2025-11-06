# Logo Images

## Cara Menambahkan Logo di Main Menu:

### Metode 1: Melalui Godot Editor (RECOMMENDED)

1. Simpan gambar logo Anda ke folder ini (PNG atau format lain)

2. Di Godot Editor:
   - Buka scene `MainMenu.tscn`
   - Di Scene tree, navigasi ke: `MainContainer → LeftSide → LogoContainer → LogoPlaceholder`
   - Di Inspector panel, cari property `Texture`
   - Drag gambar logo dari FileSystem panel dan drop ke property Texture
   - Text placeholder "[ Tempatkan Logo Di Sini ]" akan otomatis hilang

3. Sesuaikan tampilan logo:
   - `Expand Mode`: **Ignore Size** untuk ukuran penuh
   - `Stretch Mode`: **Keep Aspect Centered** agar logo tidak terdistorsi

### Metode 2: Via Script

Tambahkan di `MainMenu.gd` fungsi `_ready()`:
```gdscript
func _ready():
    print("Main Menu loaded")
    start_button.grab_focus()
    set_logo_image("res://assets/logos/app_logo.png")  # Ganti dengan path logo Anda
```

## Rekomendasi Logo:

- **Resolusi**: 512x512 px atau lebih (square) atau custom aspect ratio
- **Format**: PNG dengan transparansi untuk hasil terbaik
- **Ukuran file**: < 2 MB
- **Background**: Transparan (PNG alpha channel) untuk blending yang baik
- **Ukuran tampilan**: Logo akan ditampilkan maksimal 400x400 px di menu

## Ukuran Area Logo:

- **Area logo di kiri**: 55% dari lebar layar
- **Area menu di kanan**: 45% dari lebar layar
- **Placeholder size**: 400x400 px (dapat disesuaikan di MainMenu.tscn)

## Catatan:

- Logo akan ditampilkan di **sebelah KIRI** layar
- Menu tombol akan berada di **sebelah KANAN** layar
- Text placeholder akan hilang otomatis saat logo dimuat
- Logo akan ter-center secara vertikal dan horizontal di area kiri

## Custom Size Logo:

Jika ingin mengubah ukuran area logo, edit di `MainMenu.tscn`:
```
[node name="LogoPlaceholder" type="TextureRect" parent="MainContainer/LeftSide/LogoContainer"]
custom_minimum_size = Vector2(400, 400)  # Ubah sesuai kebutuhan
```
