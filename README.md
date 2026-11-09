# imgto3d
Arsitektur pipeline berikut dirancang untuk mengubah citra raster 2D (tunggal maupun multi-file) menjadi representasi batas solid 3D (Boundary Representation / B-Rep triangulasi) yang terkalibrasi secara metrik, kedap air (watertight), dan manifold untuk kebutuhan manufaktur (3D printing), CAD, maupun visualisasi arsitektural (SketchUp).

1. Arsitektur Matematis & Algoritmik End-to-End
   Pipeline dibagi menjadi dua jalur rekonstruksi geometris tergantung pada karakteristik citra masukan:
   - Aspek RekonstruksiMode A: Presisi CAD / OrtografisMode B: Presisi Permukaan / Relief & Objek 3D
   - Target InputDenah arsitektur, gambar teknik, vektor/logo, profil komponen 2DFoto objek tunggal/multi-sudut, tekstur relief, topografi permukaan
   - Representasi GeometriPoligon hierarkis (eksterior + lubang interior) diekstrusi pada sumbu $+Z$Calibrated 3D Point Cloud $\rightarrow$ Surface Reconstruction + Solid Closure
   - Algoritma IntiSub-pixel edge + Ramer-Douglas-Peucker + Constrained Delaunay TriangulationMonocular Depth (Depth Anything V2 / MiDaS / Heightmap) + Screened Poisson / Alpha Shapes
   - Presisi TepiSudut tajam eksak ($\epsilon$ sub-piksel)Kontinu halus dengan pelestarian fitur tajam (Taubin / Quadric Edge Collapse)

  Formulasi Matematis Setiap Tahap
  
   Normalisasi Pencahayaan & Bilateral Filtering (Tahap 1):
	Citra dikonversi ke ruang warna persepsi CIE L^* a^* b^*. Kanal luminansi L^* dinormalisasi menggunakan CLAHE (Contrast Limited Adaptive Histogram Equalization), kemudian difilter menggunakan kernel bilateral non-linear agar derau sensor hilang tanpa mengaburkan gradien tepi:
I^"filtered"  (x)=1/W_p  ∑_(x_i∈Ω)▒I (x_i ) exp⁡(-(‖I(x_i )-I(x)‖^2)/(2σ_r^2 )) exp⁡(-(‖x_i-x‖^2)/(2σ_s^2 ))
di mana σ_s mengontrol jangkauan spasial dan σ_r mengontrol ambang diskontinuitas intensitas.
	Kalibrasi Metrik Pixel-to-Real-World (Tahap 1):
Jika objek referensi di dunia nyata berukuran L_"real"  (dalam "mm" , "cm" , atau "m" ) terukur sepanjang N_"px"  piksel pada citra beresolusi W×H, faktor skala spasial s adalah:
s=L_"real" /N_"px"    ["unit/piksel" ]
Setiap koordinat piksel (u,v) dipetakan ke sistem koordinat Kartesius right-handed 2D dengan menginversi sumbu vertikal citra:
X=u⋅s, Y=(H-v)⋅s
	Ekstraksi Topologi & Rekonstruksi 3D (Tahap 2):
	Mode A (CAD/Ortografis): Kontur diekstrak secara hierarkis (RETR_CCOMP) untuk memisahkan batas luar (exterior ring, orientasi CCW) dari rongga dalam (interior holes, orientasi CW). Setiap kurva disederhanakan dengan algoritma Ramer-Douglas-Peucker menggunakan toleransi $\epsilon$ kecil agar jarak tegak lurus titik terhadap segmen memenuhi $d_{\perp}(P_k, \overline{P_1 P_n}) \le \epsilon$. Poligon divalidasi secara topologis, ditriangulasi menggunakan Constrained Delaunay Triangulation (CDT), lalu diekstrusi sejauh $H_z$ menjadi prisma solid tertutup.
	Mode B (Permukaan/Relief/3D): Peta kedalaman relatif $D(u, v) \in [0, 1]$ diestimasi melalui jaringan saraf monokular (Depth Anything V2 / MiDaS) atau pemetaan intensitas terfilter. Titik 3D diproyeksikan menjadi awan titik terkalibrasi $\mathbf{p}_i = (X_i, Y_i, Z_i)$ dengan $Z_i = D(u, v) \cdot H_z$. Normal permukaan $\hat{\mathbf{n}}_i$ diestimasi via PCA lokal ($k$-NN) dan diorientasikan secara konsisten menggunakan Minimum Spanning Tree pada bidang tangen sebelum direkonstruksi via Screened Poisson Surface Reconstruction ($\nabla \cdot \nabla \chi = \nabla \cdot \mathbf{V}$) atau Alpha Shapes.

	Optimasi Topologi & Pelestarian Sudut Tajam (Tahap 3):
Mesh dibersihkan dari degenerate faces (luas ≈0), duplicate vertices, dan komponen mengambang. Untuk meredam staircase noise tanpa menyusutkan volume CAD, diterapkan Taubin λ|μ Smoothing (λ>0,μ<-λ<0) diikuti Quadric Edge Collapse Decimation yang menjaga batas tepi tajam.

	Validasi Topologi Euler-Poincaré (Tahap 5):
Sebuah mesh poligonal tertutup dinyatakan watertight dan 2-manifold jika setiap rusuk (edge) tepat dimiliki oleh 2 segitiga dengan orientasi belitan (winding order) berlawanan serta memenuhi karakteristik Euler χ=V-E+F=2(1-g) untuk genus g.

Perintah instalasi:
python -m venv venv
# Aktivasi Linux/macOS: source venv/bin/activate
# Aktivasi Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

A. Contoh Perintah Eksekusi CLI
**Skenario 1: Mode A (Presisi CAD / Ortografis) — Denah atau Komponen Mekanikal**
Misalkan pada gambar denah_lantai1.png, dinding sepanjang $6.0\text{ meter}$ terukur sepanjang $1200\text{ piksel}$, dan Anda ingin mengekstrusi dinding setinggi $3.5\text{ meter}$ siap import ke SketchUp:Bash : 

python img2mesh_pipeline.py \
  --input denah_lantai1.png \
  --output-dir ./hasil_cad \
  --mode cad \
  --ref-real-size 6.0 \
  --ref-pixels 1200 \
  --unit m \
  --extrude-height 3.5 \
  --rdp-epsilon 0.25

  **Skenario 2: Mode A Multi-File** — Penumpukan Layer Bertingkat (Stacked Assembly)Menggabungkan 3 irisan gambar (base_plate.png, mid_wall.png, top_flange.png) menjadi satu rakitan CAD solid 3D berurutan pada sumbu $Z$:
  
  python img2mesh_pipeline.py \
  --input base_plate.png mid_wall.png top_flange.png \
  --output-dir ./hasil_assembly \
  --mode cad \
  --multi-strategy stack \
  --ref-real-size 50.0 \
  --ref-pixels 400 \
  --unit mm \
  --extrude-height 4.0

  **Skenario 3: Mode B (Presisi Permukaan / Relief & Objek 3D) ** — Foto TunggalMengonversi foto objek/relief ukiran_kayu.jpg menggunakan Depth Anything V2 dan Screened Poisson Surface Reconstruction dengan alas tertutup (watertight) setebal $15\text{ mm}$:

  python img2mesh_pipeline.py \
  --input ukiran_kayu.jpg \
  --output-dir ./hasil_relief \
  --mode surface \
  --depth-backend depth-anything \
  --surface-method poisson \
  --poisson-depth 9 \
  --ref-real-size 200.0 \
  --ref-pixels 1024 \
  --unit mm \
  --extrude-height 15.0

B. **Catatan Rekayasa & Kalibrasi Presisi**

- Integrasi Langsung ke SketchUp (.DAE): Modul _export_sketchup_collada menyuntikkan elemen <unit name="millimeter" meter="0.001"/> (atau sentimeter/meter sesuai argumen --unit) dan <up_axis>Z_UP</up_axis>. Saat diimpor ke SketchUp (File → Import → COLLADA), sumbu X–Y akan sejajar sempurna dengan bidang tanah (Red–Green plane) dan ketinggian ekstrusi menghadap lurus ke sumbu biru (+Z) dengan rasio dimensi 1:1.
	Pemilihan Nilai ϵ Ramer-Douglas-Peucker (--rdp-epsilon): Untuk gambar beresolusi tinggi (lebih dari 2000" px" ) dengan kurva melengkung atau lingkaran CAD, pertahankan --rdp-epsilon di rentang 0.20 hingga 0.40 piksel agar busur tetap mulus tanpa faset poligonal kasar.
	Koreksi Polaritas Binarization (--invert): Secara bawaan, Mode A mengasumsikan garis/objek berwarna gelap di atas latar belakang terang (seperti gambar teknik/denah putih). Gunakan flag --invert apabila objek target berwarna putih/terang di atas latar belakang gelap.


