# Panduan Menjalankan Aplikasi Mustache Filter

## Prasyarat

- **OS**: Windows 10/11
- **Python**: 3.10+
- **Godot Engine**: 4.x
- **Webcam**: Built-in atau external
- **RAM**: Minimal 4GB

---

## Instalasi

### 1. Setup Python Backend

```powershell
cd svm_orb_mustache

# Buat virtual environment
python -m venv venv

# Aktifkan virtual environment
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Setup Godot Client

1. Buka **Godot Engine 4.x**
2. Click **Import** → Navigate ke folder `godot_client`
3. Select `project.godot` → Click **Import & Edit**

---

## Menjalankan Aplikasi

### Step 1: Jalankan Backend Server

```powershell
cd svm_orb_mustache
.\venv\Scripts\Activate.ps1
python server_udp.py
```

**Output:**
```
[INFO] Models loaded successfully
[INFO] UDP Server started on ports 5005 (video) & 5006 (command)
```

### Step 2: Jalankan Godot Client

- Di Godot Editor, tekan **F5** atau click tombol **Play** (▶)
- Aplikasi akan start di Main Menu
- Click **"Mulai"** untuk memulai

---

## Penggunaan

### Kontrol Aplikasi

- **Style Buttons (1-5)**: Ubah model kumis
- **Sliders**: Fine-tuning parameter deteksi

### Parameter Tuning

| Parameter | Range | Fungsi |
|-----------|-------|--------|
| **Scale Factor** | 1.05 - 2.0 | Multi-scale detection (kecil = akurat, besar = cepat) |
| **Min Neighbors** | 1 - 10 | Validasi deteksi (besar = strict) |
| **Mustache Scale** | 0.1 - 3.0 | Ukuran kumis |
| **Mustache Y Offset** | 0.0 - 1.0 | Posisi vertikal kumis |
| **Smoothing Factor** | 0.0 - 1.0 | Mengurangi jitter |

### Rekomendasi Setting

**Balance (Default):**
- Scale Factor: 1.2
- Min Neighbors: 5
- Smoothing: 0.5

**High Accuracy:**
- Scale Factor: 1.05-1.1
- Min Neighbors: 7-10
- Smoothing: 0.6-0.8

**High Speed:**
- Scale Factor: 1.3-1.5
- Min Neighbors: 3-5
- Smoothing: 0.3-0.5

---

## Troubleshooting

### Webcam tidak terdeteksi
- Check privacy settings Windows (Camera access)
- Close aplikasi lain yang menggunakan webcam

### Wajah tidak terdeteksi
- Pastikan pencahayaan cukup
- Hadapkan wajah secara frontal
- Turunkan Scale Factor ke 1.1
- Turunkan Min Neighbors ke 3-4

### Connection refused di Godot
- Pastikan Python server sudah running
- Check firewall tidak memblokir port 5005 & 5006

### Kumis bergetar (jitter)
- Tingkatkan Smoothing Factor ke 0.7-0.9

### FPS rendah
- Tingkatkan Scale Factor ke 1.3-1.5
- Close aplikasi background yang berat

---

## Menutup Aplikasi

- **Backend**: Tekan **Ctrl+C** di terminal
- **Frontend**: Close window atau click **Keluar** di menu

---

## Port yang Digunakan

- **5005**: Video streaming (Server → Client)
- **5006**: Commands (Client → Server)

---

## Struktur Project

```
Mustache Filter/
├── svm_orb_mustache/
│   ├── server_udp.py          # Backend server (UDP)
│   ├── requirements.txt       # Python dependencies
│   ├── models/                # Trained SVM models
│   ├── assets/
│   │   ├── cascades/          # Haar Cascade XML files
│   │   └── mustaches/         # Mustache PNG assets
│   └── pipelines/             # Core processing modules
│
└── godot_client/
    ├── project.godot          # Godot project file
    ├── scenes/
    │   ├── MainMenu.tscn      # Main menu scene
    │   └── Main.tscn          # Application scene
    └── scripts/
        ├── UDPReceiver.gd     # UDP communication handler
        └── VideoDisplay.gd    # Video rendering
```
