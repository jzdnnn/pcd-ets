# Background Images

## Cara Menambahkan Background Image untuk Main Menu:

### PENTING: Layout Main Menu
Main menu menggunakan layout split:
- **Sebelah KIRI (55% lebar)**: Space kosong untuk logo
- **Sebelah KANAN (45% lebar)**: Menu tombol

**Anda harus menempelkan logo langsung di background image** di sebelah kiri sebelum mengimportnya ke Godot.

## Cara Membuat Background dengan Logo:

1. **Gunakan software image editor** (Photoshop, GIMP, Canva, dll)
2. **Buat canvas**: 1920x1080 px (Full HD)
3. **Tempatkan logo di sebelah KIRI** (area sekitar 1056x1080 px)
4. **Sisakan area KANAN** untuk menu (area sekitar 864x1080 px)
5. **Export** sebagai PNG atau JPG
6. **Simpan** ke folder ini (`godot_client/assets/backgrounds/`)

## Layout Guide untuk Image Editor:

```
┌─────────────────────────────────────────────────┐
│          1920 x 1080 px                         │
│                                                 │
│  ┌──────────────────┬──────────────────────┐   │
│  │                  │                      │   │
│  │   LOGO AREA      │    MENU AREA        │   │
│  │   (1056 px)      │    (864 px)         │   │
│  │                  │    [KOSONG]         │   │
│  │  ┌──────────┐    │                      │   │
│  │  │          │    │   Menu tombol akan   │   │
│  │  │   LOGO   │    │   muncul di sini     │   │
│  │  │          │    │   dari Godot         │   │
│  │  └──────────┘    │                      │   │
│  │                  │                      │   │
│  │  + Background    │   + Background       │   │
│  │    design        │     design           │   │
│  └──────────────────┴──────────────────────┘   │
└─────────────────────────────────────────────────┘
```

## Cara Import ke Godot:

1. **Simpan background** (dengan logo) ke folder ini
2. **Buka Godot Editor** dan load project
3. **Di FileSystem panel**, refresh untuk melihat gambar baru
4. **Buka scene** `MainMenu.tscn`
5. **Pilih node**: `BackgroundContainer → BackgroundImage`
6. **Di Inspector**: Drag gambar ke property **Texture**
7. **Save** scene (Ctrl+S)

## Rekomendasi:

- **Resolusi**: 1920x1080 px (Full HD) atau 1280x720 px (HD)
- **Format**: PNG (untuk kualitas terbaik) atau JPG (untuk ukuran lebih kecil)
- **Ukuran file**: Maksimal 5 MB untuk performa optimal
- **Logo position**: Sebelah KIRI (dalam area 1056x1080 px)
- **Menu area**: Sebelah KANAN (biarkan kosong atau background sederhana, area 864x1080 px)
- **Overlay**: Godot akan menambahkan overlay gelap 40% otomatis

## Tips Design:

1. **Logo di kiri**: Center vertical dan horizontal dalam area kiri
2. **Background kanan**: Gunakan warna/pattern yang tidak mengganggu teks menu
3. **Kontras**: Pastikan area kanan cukup gelap agar teks putih terbaca
4. **Testing**: Test dengan resolution berbeda (windowed, fullscreen)

## Catatan:

- Background sudah dikonfigurasi dengan `expand_mode = 1` dan `stretch_mode = 5` (Scale)
- Overlay gelap 40% akan ditambahkan otomatis di atas background
- Area kiri (55%) akan kosong jika tidak ada background, menampilkan warna solid
- Menu tombol akan muncul di area kanan (45%) secara otomatis dari Godot
