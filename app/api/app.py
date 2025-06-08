"""
Plant Disease Classification API

This FastAPI application provides an API for classifying plant diseases using a Keras model.
It includes endpoints for health checks and image prediction with proper error handling.
Enhanced with plant/leaf detection to validate that uploaded images contain plant leaves.
"""

import os
import io
import traceback
import logging
from typing import List, Tuple, Dict, Any, Optional
import mimetypes
import imghdr
import cv2
import numpy as np
import tensorflow as tf

from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, UploadFile, HTTPException, status, Depends, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join("keras_model", "best_model.h5")

DEFAULT_INPUT_SIZE = (224, 224)

CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

INDONESIAN_TRANSLATIONS = {
    "Apple___Apple_scab": "Apel - Kudis Apel",
    "Apple___Black_rot": "Apel - Busuk Hitam",
    "Apple___Cedar_apple_rust": "Apel - Karat Cedar",
    "Apple___healthy": "Apel - Sehat",
    "Blueberry___healthy": "Blueberry - Sehat",
    "Cherry_(including_sour)___Powdery_mildew": "Ceri - Embun Tepung",
    "Cherry_(including_sour)___healthy": "Ceri - Sehat",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Jagung - Bercak Daun Cercospora",
    "Corn_(maize)___Common_rust_": "Jagung - Karat Biasa",
    "Corn_(maize)___Northern_Leaf_Blight": "Jagung - Hawar Daun Utara",
    "Corn_(maize)___healthy": "Jagung - Sehat",
    "Grape___Black_rot": "Anggur - Busuk Hitam",
    "Grape___Esca_(Black_Measles)": "Anggur - Esca (Campak Hitam)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Anggur - Hawar Daun (Bercak Isariopsis)",
    "Grape___healthy": "Anggur - Sehat",
    "Orange___Haunglongbing_(Citrus_greening)": "Jeruk - Huanglongbing (Penghijauan Sitrus)",
    "Peach___Bacterial_spot": "Persik - Bercak Bakteri",
    "Peach___healthy": "Persik - Sehat",
    "Pepper,_bell___Bacterial_spot": "Paprika - Bercak Bakteri",
    "Pepper,_bell___healthy": "Paprika - Sehat",
    "Potato___Early_blight": "Kentang - Hawar Awal",
    "Potato___Late_blight": "Kentang - Hawar Akhir",
    "Potato___healthy": "Kentang - Sehat",
    "Raspberry___healthy": "Raspberry - Sehat",
    "Soybean___healthy": "Kedelai - Sehat",
    "Squash___Powdery_mildew": "Labu - Embun Tepung",
    "Strawberry___Leaf_scorch": "Stroberi - Gosong Daun",
    "Strawberry___healthy": "Stroberi - Sehat",
    "Tomato___Bacterial_spot": "Tomat - Bercak Bakteri",
    "Tomato___Early_blight": "Tomat - Hawar Awal",
    "Tomato___Late_blight": "Tomat - Hawar Akhir",
    "Tomato___Leaf_Mold": "Tomat - Jamur Daun",
    "Tomato___Septoria_leaf_spot": "Tomat - Bercak Daun Septoria",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Tomat - Tungau Laba-laba",
    "Tomato___Target_Spot": "Tomat - Bercak Target",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Tomat - Virus Keriting Daun Kuning",
    "Tomato___Tomato_mosaic_virus": "Tomat - Virus Mosaik Tomat",
    "Tomato___healthy": "Tomat - Sehat"
}

TREATMENT_SUGGESTIONS = {
    "Apple___Apple_scab": """
    Kudis apel disebabkan oleh jamur Venturia inaequalis yang berkembang dalam kondisi lembab dan suhu 15-25°C. Gejala awal berupa bercak hijau gelap pada daun yang kemudian berubah menjadi coklat kehitaman dengan tepi yang tidak beraturan. Pada buah, muncul bercak hitam yang dapat menyebabkan buah retak dan cacat.

    Untuk pengendalian, aplikasikan fungisida berbahan aktif captan dengan konsentrasi 0.2-0.3% atau mancozeb 0.2% setiap 7-10 hari, terutama saat cuaca lembab. Mulai penyemprotan sejak tunas daun mulai membuka hingga 2 minggu setelah kelopak bunga gugur. Tingkatkan frekuensi menjadi setiap 5-7 hari jika curah hujan tinggi.

    Lakukan sanitasi kebun dengan membuang semua daun gugur yang terinfeksi dan membakarnya atau mengomposkannya dengan suhu tinggi (>60°C). Pangkas cabang yang terinfeksi dengan memotong 15-20 cm di bawah area yang terinfeksi. Sterilkan alat pemangkas dengan larutan klorin 10% atau alkohol 70% setiap kali digunakan.

    Perbaiki sirkulasi udara dengan mengatur jarak tanam minimal 4-5 meter antar pohon dan memangkas cabang yang terlalu rapat, terutama di bagian tengah kanopi. Hindari penyiraman overhead dan gunakan sistem irigasi tetes untuk mengurangi kelembaban daun. Aplikasikan mulsa organik setebal 7-10 cm di sekitar pangkal pohon untuk mencegah percikan tanah yang mengandung spora jamur ke daun.
    """,
    
    "Apple___Black_rot": """
    Busuk hitam apel disebabkan oleh jamur Botryosphaeria obtusa yang menyerang buah, daun, dan dapat menyebabkan kanker pada batang. Gejala pada buah dimulai dengan bercak coklat kecil yang berkembang menjadi bercak besar dengan pola lingkaran konsentris. Buah yang terinfeksi parah akan mengkerut dan berubah warna menjadi hitam kecoklatan.

    Sanitasi merupakan kunci utama pengendalian. Buang semua buah yang menunjukkan gejala infeksi, baik yang masih menggantung di pohon maupun yang sudah jatuh ke tanah. Lakukan pemeriksaan rutin setiap 3-4 hari selama musim buah. Potong dan musnahkan semua bagian tanaman yang terinfeksi, termasuk ranting dan daun yang menunjukkan gejala kanker atau bercak.

    Aplikasikan fungisida berbahan aktif captan 50 WP dengan dosis 2-3 gram per liter air setiap 10-14 hari, dimulai dari fase pembungaan hingga menjelang panen. Pada kondisi cuaca yang sangat lembab, interval penyemprotan dapat diperpendek menjadi 7-10 hari. Pastikan coverage yang merata pada seluruh bagian tanaman, terutama buah yang sedang berkembang.

    Cegah terjadinya luka pada buah dengan menghindari pemanenan kasar dan melindungi dari kerusakan akibat hama serangga atau burung. Perbaiki drainase tanah untuk mencegah genangan air yang dapat meningkatkan kelembaban dan mempercepat perkembangan penyakit. Lakukan pemangkasan untuk membuka kanopi dan meningkatkan penetrasi sinar matahari serta sirkulasi udara.
    """,
    
    "Apple___Cedar_apple_rust": """
    Karat cedar-apel adalah penyakit yang disebabkan oleh jamur Gymnosporangium juniperi-virginianae yang membutuhkan dua inang untuk menyelesaikan siklus hidupnya: pohon cedar/juniper dan pohon apel. Gejala pada apel berupa bercak kuning-oranye pada daun dengan struktur seperti tanduk di bagian bawah daun.

    Pengendalian kimiawi menggunakan fungisida berbahan aktif myclobutanil atau propiconazole dengan konsentrasi 0.1-0.15%. Aplikasi pertama dilakukan saat tunas mulai membengkak (sebelum daun muncul), kemudian diulang setiap 10-14 hari hingga 4-6 kali aplikasi. Timing yang tepat sangat krusial karena infeksi terjadi pada fase awal pertumbuhan daun.

    Jika memungkinkan, hilangkan pohon cedar atau juniper dalam radius 1-2 km dari kebun apel, karena spora jamur dapat terbawa angin sejauh tersebut. Jika tidak memungkinkan, fokuskan pada aplikasi fungisida preventif yang konsisten. Buang dan musnahkan daun apel yang terinfeksi untuk mengurangi sumber inokulum.

    Pilih varietas apel yang tahan terhadap karat cedar seperti Enterprise, Liberty, atau Pristine. Lakukan pemangkasan untuk meningkatkan sirkulasi udara dan mengurangi periode basah pada daun. Monitor kondisi cuaca dan tingkatkan frekuensi aplikasi fungisida saat periode basah yang berkepanjangan, terutama pada suhu 18-24°C yang optimal untuk perkembangan penyakit.
    """,
    
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": """
    Bercak daun Cercospora pada jagung disebabkan oleh jamur Cercospora zeae-maydis yang berkembang pada kondisi hangat dan lembab. Gejala berupa bercak persegi panjang berwarna abu-abu hingga coklat dengan tepi yang tajam, sejajar dengan tulang daun. Infeksi berat dapat menyebabkan daun mengering dan mati prematur.

    Pengendalian fungisida menggunakan bahan aktif azoxystrobin atau pyraclostrobin dengan dosis sesuai anjuran (biasanya 0.5-1 ml/liter). Aplikasi pertama dilakukan saat gejala awal muncul atau secara preventif pada fase vegetatif akhir (V12-VT). Ulangi aplikasi 14-21 hari kemudian jika kondisi cuaca mendukung perkembangan penyakit.

    Implementasikan rotasi tanaman dengan tanaman non-graminae seperti kedelai, kacang tanah, atau sayuran selama minimal 2-3 musim tanam. Ini akan memutus siklus hidup patogen yang dapat bertahan pada sisa tanaman jagung. Bersihkan dan bakar sisa tanaman jagung setelah panen untuk mengurangi sumber inokulum untuk musim berikutnya.

    Atur jarak tanam yang optimal (70-75 cm antar baris, 20-25 cm dalam baris) untuk memastikan sirkulasi udara yang baik dan mengurangi kelembaban di sekitar tanaman. Hindari penanaman terlalu rapat yang dapat menciptakan mikroklimat lembab yang kondusif untuk perkembangan jamur. Lakukan pemupukan berimbang dengan menghindari nitrogen berlebihan yang membuat tanaman lebih rentan terhadap penyakit.
    """,
    
    "Corn_(maize)___Common_rust_": """
    Karat jagung biasa disebabkan oleh jamur Puccinia sorghi yang menghasilkan pustula karat berwarna coklat kemerahan pada kedua sisi daun. Penyakit ini berkembang pada suhu 16-25°C dengan kelembaban tinggi dan dapat menyebabkan penurunan hasil hingga 10-40% jika infeksi terjadi pada fase kritis.

    Penggunaan varietas tahan karat merupakan strategi pengendalian paling efektif dan ekonomis. Pilih varietas yang memiliki gen ketahanan Rp1, Rp3, atau kombinasi gen ketahanan lainnya. Konsultasikan dengan penangkar benih lokal untuk mendapatkan varietas yang sesuai dengan kondisi wilayah dan memiliki ketahanan terhadap ras karat yang dominan di daerah tersebut.

    Jika gejala mulai muncul, aplikasikan fungisida berbahan aktif azoxystrobin atau tebuconazole dengan interval 14-21 hari. Mulai aplikasi saat 5-10% tanaman menunjukkan gejala awal atau secara preventif saat kondisi cuaca sangat mendukung (suhu dingin dan kelembaban tinggi). Pastikan coverage yang baik pada bagian bawah daun dimana pustula sering kali muncul pertama kali.

    Terapkan jarak tanam yang sesuai rekomendasi untuk memfasilitasi sirkulasi udara yang optimal. Lakukan rotasi tanaman dengan legum atau tanaman lain selain graminae untuk memutus siklus penyakit. Monitor kondisi cuaca dan tingkatkan kewaspadaan saat terjadi embun pagi yang berkepanjangan atau hujan ringan yang sering, karena ini menciptakan kondisi ideal untuk infeksi spora karat.
    """,
    
    "Corn_(maize)___Northern_Leaf_Blight": """
    Hawar daun utara jagung disebabkan oleh jamur Exserohilum turcicum (sinonim: Helminthosporium turcicum) yang menghasilkan bercak elips memanjang berwarna abu-abu hingga coklat dengan panjang 2.5-15 cm. Penyakit ini berkembang optimal pada suhu 18-27°C dengan kelembaban relatif tinggi (>90%).

    Aplikasi fungisida dilakukan dengan bahan aktif azoxystrobin, pyraclostrobin, atau kombinasi strobilurin dengan triazole. Konsentrasi yang digunakan adalah 0.5-1 ml/liter air dengan volume semprot 400-600 liter per hektar. Timing aplikasi sangat penting: mulai saat tanaman mencapai fase V8-V10 atau ketika gejala pertama muncul pada daun bagian bawah, kemudian diulang 14-21 hari kemudian.

    Sanitasi lahan dengan membersihkan dan menghancurkan sisa tanaman jagung setelah panen sangat penting karena jamur dapat bertahan hidup pada debris tanaman hingga 2-3 tahun. Lakukan pengolahan tanah yang baik untuk mengubur sisa tanaman dan mempercepat dekomposisi. Rotasi dengan tanaman bukan graminae selama minimal 2 musim akan sangat efektif mengurangi populasi patogen.

    Pengelolaan populasi tanaman dengan mengatur jarak tanam optimal (70-80 cm antar barisan, 20-25 cm dalam barisan) untuk mengurangi kelembaban mikro di sekitar tanaman. Lakukan pemangkasan daun bagian bawah yang sudah tua atau mulai menguning untuk meningkatkan sirkulasi udara. Hindari pemupukan nitrogen berlebihan yang dapat membuat tanaman lebih sukulen dan rentan terhadap infeksi.
    """,
    
    "Grape___Black_rot": """
    Busuk hitam anggur disebabkan oleh jamur Guignardia bidwellii yang menyerang buah, daun, dan tunas muda. Pada buah, gejala dimulai dengan bercak coklat kecil yang berkembang menjadi busuk hitam dengan buah mengkerut seperti kismis. Pada daun muncul bercak coklat dengan tepi gelap.

    Pengendalian fungisida menggunakan mancozeb 80 WP dengan dosis 2-2.5 gram per liter air atau captan 50 WP dengan dosis yang sama. Aplikasi dimulai sejak tunas mulai pecah (bud break) dan diulang setiap 10-14 hari hingga buah mencapai ukuran kacang polong. Pada periode kritis (pembungaan hingga fruit set), interval dapat diperpendek menjadi 7-10 hari jika cuaca sangat lembab.

    Sanitasi kebun dilakukan dengan membuang semua buah yang terinfeksi (termasuk buah mumi yang menggantung), daun yang bergejala, dan tunas yang terinfeksi. Material yang terinfeksi harus dibakar atau dikubur dalam-dalam, jangan dijadikan kompos. Bersihkan area di bawah tanaman dari daun gugur yang dapat menjadi sumber inokulum untuk musim berikutnya.

    Manajemen kanopi dengan pemangkasan yang tepat untuk membuka struktur tanaman dan meningkatkan penetrasi udara serta sinar matahari. Lakukan thinning cluster untuk mengurangi kepadatan buah dan meningkatkan sirkulasi udara di antara buah. Hindari penyiraman overhead dan gunakan sistem irigasi tetes. Kontrol gulma di sekitar tanaman untuk mengurangi kelembaban lokal.
    """,
    
    "Grape___Esca_(Black_Measles)": """
    Esca atau campak hitam anggur adalah penyakit kompleks yang disebabkan oleh beberapa jamur patogen termasuk Phaeomoniella chlamydospora, Phaeoacremonium spp., dan Fomitiporia mediterranea. Gejala berupa hawar daun dengan pola khas seperti harimau, buah berbercak, dan pembusukan kayu pada batang tua.

    Penanganan dimulai dengan pemotongan bagian yang terinfeksi. Potong cabang atau batang yang menunjukkan gejala pembusukan kayu hingga mencapai jaringan sehat (biasanya 30-50 cm di bawah area yang terlihat terinfeksi). Waktu pemangkasan terbaik adalah saat musim kering untuk meminimalkan risiko infeksi baru. Sterilkan alat potong dengan alkohol 70% atau larutan klorin 10% setiap kali digunakan.

    Aplikasi fungisida sistemik berbahan aktif fosetyl-al dengan dosis 2-3 gram per liter air melalui penyemprotan daun atau injeksi batang. Untuk injeksi batang, lakukan pada awal musim semi saat aliran getah mulai aktif. Penyemprotan daun dilakukan setiap 3-4 minggu selama musim tumbuh, terutama pada tanaman yang menunjukkan gejala ringan.

    Pencegahan luka pada tanaman sangat penting karena jamur masuk melalui luka pemangkasan atau kerusakan mekanis. Gunakan pasta pelindung berbahan aktif fungisida pada bekas luka pemangkasan besar. Perbaiki drainase tanah untuk menghindari stres air yang dapat melemahkan ketahanan tanaman. Hindari pemupukan nitrogen berlebihan yang dapat membuat tanaman lebih rentan.
    """,
    
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": """
    Hawar daun anggur yang disebabkan oleh Isariopsis clavispora menghasilkan bercak coklat tidak beraturan pada daun dengan sporulasi berwarna gelap di bagian bawah daun. Penyakit ini berkembang pada kondisi hangat dan lembab, terutama pada akhir musim panas.

    Pengendalian fungisida menggunakan mancozeb atau chlorothalonil dengan konsentrasi 0.2-0.25%. Aplikasi dimulai saat gejala pertama muncul atau secara preventif pada periode rentan (suhu 25-30°C dengan kelembaban tinggi). Interval aplikasi 10-14 hari, dapat diperpendek menjadi 7-10 hari pada kondisi cuaca yang sangat mendukung perkembangan penyakit.

    Buang dan musnahkan daun yang terinfeksi untuk mengurangi sumber inokulum. Lakukan sanitasi rutin dengan membersihkan daun gugur di area sekitar tanaman. Pastikan pembuangan material terinfeksi jauh dari kebun atau dibakar untuk mencegah penyebaran spora. Monitoring dilakukan secara rutin, terutama pada bagian bawah kanopi yang cenderung lebih lembab.

    Pengelolaan lingkungan dengan meningkatkan sirkulasi udara melalui pemangkasan yang tepat dan pengaturan sistem trellis yang memungkinkan aliran udara optimal. Hindari penyiraman overhead terutama pada sore hari yang dapat menyebabkan daun basah sepanjang malam. Gunakan mulsa organik untuk mengurangi percikan tanah yang dapat membawa spora ke daun bagian bawah.
    """,
    
    "Orange___Haunglongbing_(Citrus_greening)": """
    Huanglongbing (HLB) atau citrus greening disebabkan oleh bakteri Candidatus Liberibacter yang ditularkan oleh serangga psyllid (Diaphorina citri). Gejala berupa menguningnya daun dengan pola asimetris, buah kecil dan asam, serta penurunan produksi yang drastis.

    Pengendalian vektor psyllid merupakan kunci utama pencegahan. Aplikasikan insektisida sistemik berbahan aktif imidacloprid atau thiamethoxam melalui soil drench dengan dosis 0.5-1 gram per liter air, diberikan di sekitar zona perakaran. Untuk penyemprotan foliar, gunakan insektisida kontak seperti malathion atau lambda-cyhalothrin setiap 2-3 minggu, terutama saat flush daun muda muncul.

    Deteksi dini dan eradikasi tanaman terinfeksi sangat penting untuk mencegah penyebaran. Lakukan survei rutin setiap 2-3 bulan dengan memperhatikan gejala khas pada daun dan buah. Tanaman yang positif terinfeksi harus segera dicabut dan dimusnahkan dengan cara dibakar atau dikubur dalam-dalam. Laporkan ke dinas pertanian setempat untuk koordinasi pengendalian area.

    Program monitoring berkelanjutan dengan pemasangan perangkap kuning lengket untuk memantau populasi psyllid dewasa. Lakukan pemeriksaan visual rutin pada flush daun muda yang merupakan tempat favorit psyllid untuk bertelur dan makan. Koordinasi dengan petani tetangga untuk pengendalian terpadu karena psyllid dapat terbang antar kebun dalam radius beberapa kilometer.
    """,
    
    "Peach___Bacterial_spot": """
    Bercak bakteri pada peach disebabkan oleh Xanthomonas arboricola pv. pruni yang menyerang daun, buah, dan tunas. Gejala pada daun berupa bercak kecil bersudut dengan halo kuning, sedangkan pada buah muncul bercak cekung yang dapat berkembang menjadi retakan.

    Pemilihan varietas tahan merupakan strategi jangka panjang yang paling efektif. Varietas seperti 'Candor', 'Harmony', atau 'Clayton' menunjukkan ketahanan yang baik terhadap bacterial spot. Konsultasikan dengan nursery atau dinas pertanian setempat untuk mendapatkan varietas tahan yang sesuai dengan kondisi iklim lokal.

    Aplikasi bakterisida berbahan aktif tembaga seperti copper hydroxide atau copper sulfate dengan konsentrasi 0.1-0.15%. Penyemprotan dimulai saat tunas mulai membengkak dan diulang setiap 10-14 hari hingga 2-3 minggu sebelum panen. Hindari aplikasi saat suhu sangat tinggi (>30°C) untuk mencegah fitotoksisitas. Tambahkan sticker-spreader untuk meningkatkan adhesi dan persistensi bakterisida.

    Pengelolaan irigasi dengan menghindari penyiraman overhead yang dapat menyebarkan bakteri melalui percikan air. Gunakan sistem irigasi tetes atau sprinkler micro yang mengarah ke tanah. Lakukan penyiraman pada pagi hari agar tanaman cepat kering. Tingkatkan sirkulasi udara dengan pemangkasan yang membuka struktur kanopi dan mengurangi periode basah pada daun.
    """,
    
    "Pepper,_bell___Bacterial_spot": """
    Bercak bakteri pada paprika disebabkan oleh kompleks bakteri Xanthomonas spp. yang menghasilkan bercak coklat dengan halo kuning pada daun dan bercak cekung pada buah. Infeksi dapat menyebabkan defoliasi dan penurunan kualitas buah yang signifikan.

    Penggunaan varietas tahan seperti yang memiliki gen Bs2 atau Bs3 memberikan proteksi yang baik terhadap beberapa strain bakteri. Pilih benih dari sumber terpercaya dan lakukan seed treatment dengan bakterisida berbahan aktif streptomycin atau copper compound sebelum semai. Rendam benih dalam larutan bakterisida selama 30 menit, kemudian keringkan sebelum disemai.

    Program aplikasi bakterisida menggunakan copper compound (copper hydroxide atau copper sulfate) dengan konsentrasi 0.1-0.2%. Aplikasi dimulai sejak fase semai di persemaian dan dilanjutkan setiap 7-10 hari setelah tanam di lapangan. Pada kondisi cuaca lembab atau setelah hujan, frekuensi dapat ditingkatkan. Rotasikan dengan bakterisida berbahan aktif streptomycin untuk mencegah resistensi.

    Implementasi rotasi tanaman dengan family non-Solanaceae seperti brassica, legum, atau cucurbit selama minimal 2-3 musim. Sanitasi lahan dengan membersihkan sisa tanaman dan gulma yang dapat menjadi inang alternatif. Atur jarak tanam optimal (40-50 cm dalam baris, 60-70 cm antar baris) dan gunakan mulsa plastik untuk mengurangi percikan tanah yang mengandung bakteri ke daun bagian bawah.
    """,
    
    "Potato___Early_blight": """
    Hawar awal kentang disebabkan oleh jamur Alternaria solani yang menghasilkan bercak coklat konsentris pada daun, dimulai dari daun tua di bagian bawah tanaman. Infeksi dapat menyebar ke umbi dan menyebabkan busuk kering dengan bercak cekung berwarna coklat gelap.

    Pengendalian fungisida menggunakan mancozeb 80 WP dengan dosis 2-2.5 gram per liter air atau chlorothalonil 75 WP dengan dosis yang sama. Aplikasi pertama dilakukan secara preventif saat tanaman berumur 4-6 minggu atau ketika gejala pertama muncul. Interval aplikasi 7-10 hari selama periode rentan (cuaca hangat dan lembab), dapat diperpanjang menjadi 10-14 hari pada kondisi kering.

    Rotasi tanaman dengan family non-Solanaceae sangat efektif karena jamur Alternaria dapat bertahan pada sisa tanaman hingga 2-3 tahun. Rotasi dengan serealia, legum, atau brassica selama minimal 3-4 musim akan mengurangi populasi patogen secara signifikan. Pengolahan tanah yang baik untuk mengubur sisa tanaman dan mempercepat dekomposisi juga membantu mengurangi sumber inokulum.

    Pengelolaan populasi dan lingkungan dengan mengatur jarak tanam yang tidak terlalu rapat (30-35 cm dalam baris, 70-80 cm antar baris) untuk meningkatkan sirkulasi udara. Lakukan pembumbunan yang tepat untuk mengurangi paparan umbi terhadap spora jamur. Hindari penyiraman overhead pada sore hari dan pastikan drainase yang baik untuk mengurangi kelembaban tanah yang berlebihan.
    """,
    
    "Potato___Late_blight": """
    Hawar akhir kentang disebabkan oleh Phytophthora infestans, patogen yang sama penyebab bencana kelaparan kentang di Irlandia. Gejala berupa bercak basah berwarna coklat gelap pada daun dengan sporulasi putih di bagian bawah daun pada kondisi lembab. Pada umbi, terjadi busuk coklat yang dimulai dari mata tunas.

    Aplikasi fungisida sistemik berbahan aktif metalaxyl-M atau dimethomorph dikombinasikan dengan fungisida kontak seperti mancozeb atau chlorothalonil. Gunakan produk kombinasi seperti metalaxyl-M + mancozeb dengan dosis sesuai anjuran (biasanya 2-2.5 gram/liter). Aplikasi dimulai secara preventif saat kondisi cuaca mendukung (suhu 15-25°C, kelembaban >90%) dan diulang setiap 5-7 hari selama periode kritis.

    Rotasi tanaman minimal 4-5 tahun dengan tanaman bukan Solanaceae karena P. infestans dapat bertahan pada umbi yang tertinggal di tanah. Pilih lahan dengan drainase baik dan hindari penanaman di area yang sering tergenang. Gunakan bibit bebas penyakit dan lakukan sortasi ketat untuk mengeliminasi umbi yang menunjukkan gejala busuk.

    Monitoring cuaca intensif dengan menggunakan sistem peringatan dini seperti model Blitecast atau aplikasi weather monitoring yang dapat memprediksi kondisi kondusif untuk infeksi P. infestans. Tingkatkan frekuensi aplikasi fungisida saat periode Blight Units tinggi. Lakukan panen pada cuaca kering dan biarkan umbi mengering di lapangan sebelum disimpan untuk mengurangi risiko infeksi laten.
    """,
    
    "Squash___Powdery_mildew": """
    Embun tepung pada labu disebabkan oleh jamur Podosphaera xanthii yang menghasilkan lapisan putih seperti tepung pada permukaan daun. Infeksi dimulai pada daun tua dan menyebar ke seluruh tanaman, menyebabkan menguningnya daun dan penurunan fotosintesis.

    Pengendalian fungisida menggunakan sulfur elemental atau wettable sulfur dengan konsentrasi 0.2-0.3%. Aplikasi dimulai saat gejala pertama muncul atau secara preventif pada kondisi cuaca yang mendukung (suhu 20-30°C, kelembaban sedang). Hindari aplikasi sulfur pada suhu >32°C karena dapat menyebabkan fitotoksisitas. Alternatif lain adalah penggunaan bicarbonate potassium atau fungisida sistemik berbahan aktif myclobutanil.

    Pengelolaan lingkungan dengan meningkatkan sirkulasi udara melalui pengaturan jarak tanam yang optimal (1-1.5 meter antar tanaman) dan pemangkasan daun tua yang menunjukkan gejala awal. Gunakan sistem trellis atau ajir untuk meningkatkan eksposur daun terhadap sinar matahari dan aliran udara. Hindari penanaman yang terlalu rapat yang menciptakan mikroklimat lembab.

    Pengendalian kelembaban dengan menghindari penyiraman overhead dan menggunakan irigasi tetes atau furrow irrigation. Lakukan penyiraman pada pagi hari agar tanaman cepat kering. Aplikasikan mulsa organik untuk mengurangi fluktuasi kelembaban tanah dan mencegah percikan tanah ke daun. Rotasi dengan tanaman family lain dan pembersihan sisa tanaman setelah panen untuk mengurangi sumber inokulum musim berikutnya.
    """,
    
    "Strawberry___Leaf_scorch": """
    Leaf scorch pada strawberry disebabkan oleh jamur Diplocarpon earlianum yang menghasilkan bercak ungu kemerahan pada daun dengan tepi coklat. Infeksi berat dapat menyebabkan daun mengering dan mati, mengurangi kapasitas fotosintesis dan vigor tanaman.

    Aplikasi fungisida berbahan aktif captan 50 WP dengan dosis 2-2.5 gram per liter air atau myclobutanil dengan konsentrasi sesuai anjuran. Penyemprotan dimulai pada awal musim pertumbuhan dan diulang setiap 10-14 hari, terutama selama periode cuaca lembab. Pastikan coverage yang baik pada bagian bawah daun dimana gejala sering kali muncul pertama kali.

    Sanitasi kebun dengan membuang daun yang terinfeksi secara rutin, terutama daun tua yang menunjukkan gejala berat. Lakukan pembersihan menyeluruh pada akhir musim dengan memotong semua daun lama dan membakar atau mengomposkannya dengan suhu tinggi. Ganti mulsa lama dengan yang baru untuk mengurangi sumber inokulum yang dapat bertahan di permukaan tanah.

    Pengelolaan irigasi dengan menghindari penyiraman overhead yang dapat membasahi daun dalam waktu lama. Gunakan sistem irigasi tetes dengan timer yang memungkinkan penyiraman pada pagi hari sehingga tanaman cepat kering. Atur jarak tanam yang memadai (30-40 cm antar tanaman) dan lakukan penjarangan stolon untuk meningkatkan sirkulasi udara di antara tanaman.
    """,
    
    "Tomato___Bacterial_spot": """
    Bacterial spot pada tomat disebabkan oleh bakteri Xanthomonas campestris pv. vesicatoria yang menimbulkan bercak kecil berwarna coklat gelap pada daun, batang, dan buah. Infeksi berat dapat menyebabkan kerusakan jaringan dan penurunan hasil panen secara signifikan.

    Pengendalian dilakukan dengan menggunakan varietas tomat yang tahan terhadap bakteri ini. Aplikasikan bakterisida yang mengandung tembaga, seperti copper hydroxide, dengan dosis sesuai anjuran, terutama saat kondisi cuaca lembab dan basah. Penyemprotan rutin setiap 7-10 hari membantu mengurangi penyebaran penyakit.

    Hindari penyiraman dari atas yang dapat memercikkan bakteri ke bagian tanaman lain. Gunakan sistem irigasi tetes agar kelembaban daun terjaga rendah. Lakukan rotasi tanaman dengan tanaman non-solanaceae minimal setiap 2-3 musim tanam untuk memutus siklus hidup patogen.
""",

"Tomato___Early_blight": """
    Early blight disebabkan oleh jamur Alternaria solani yang menghasilkan bercak bercincin pada daun tua, batang, dan buah. Gejala awal berupa bercak coklat dengan pola konsentris yang dapat menyebabkan daun mengering dan gugur.

    Fungisida berbahan aktif mancozeb efektif digunakan sebagai kontrol penyakit ini. Penyemprotan dilakukan sejak fase awal pertumbuhan dan diulang setiap 7-10 hari, terutama selama musim hujan atau kelembaban tinggi.

    Rotasi tanaman penting untuk mengurangi keberadaan inokulum jamur di tanah. Hindari penanaman yang terlalu rapat agar sirkulasi udara tetap baik dan kelembaban daun tidak berlebih. Buang dan musnahkan daun yang sudah terinfeksi untuk mencegah penyebaran lebih lanjut.
""",

"Tomato___Late_blight": """
    Late blight merupakan penyakit serius yang disebabkan oleh oomycete Phytophthora infestans. Ditandai dengan bercak gelap berair pada daun dan buah, dengan miselium putih berbulu di bawah daun saat kondisi lembab.

    Gunakan fungisida berbahan aktif metalaxyl atau campuran metalaxyl-mancozeb sesuai anjuran label. Aplikasi fungisida harus dimulai sebelum gejala muncul dan diulang setiap 7-10 hari selama kondisi lingkungan mendukung perkembangan penyakit.

    Rotasi tanaman dan menjaga jarak tanam yang baik mengurangi kelembaban dan membantu mencegah infeksi. Buang dan musnahkan tanaman yang sudah terinfeksi untuk menghindari penyebaran patogen.
""",

"Tomato___Leaf_Mold": """
    Leaf mold disebabkan oleh jamur Passalora fulva yang menghasilkan bercak kuning pada permukaan atas daun dan lapisan jamur berwarna abu-abu kehijauan di bawah daun.

    Fungisida berbahan aktif chlorothalonil dapat digunakan untuk mengendalikan penyakit ini. Penyemprotan rutin dilakukan selama periode kelembaban tinggi, terutama jika sirkulasi udara di kebun kurang baik.

    Pastikan sirkulasi udara di kebun cukup dengan mengatur jarak tanam dan memangkas daun yang terlalu rapat. Hindari kelembaban berlebih dengan mengatur irigasi dan menggunakan rotasi tanaman sebagai langkah pencegahan jangka panjang.
""",

"Tomato___Septoria_leaf_spot": """
    Septoria leaf spot disebabkan oleh jamur Septoria lycopersici, yang memunculkan bercak kecil bulat berwarna coklat dengan titik hitam di tengahnya pada daun tomat. Infeksi parah dapat menyebabkan daun menguning dan gugur.

    Pengendalian dengan fungisida berbahan aktif mancozeb sangat dianjurkan, dengan penyemprotan dilakukan secara rutin. Sanitasi dengan membuang daun yang terinfeksi dan menghindari penyiraman dari atas membantu mengurangi penyebaran.

    Rotasi tanaman juga penting untuk menghilangkan sumber inokulum. Pengaturan jarak tanam dan sistem irigasi tetes mengurangi kelembaban pada daun yang mempercepat infeksi.
""",

"Tomato___Spider_mites Two-spotted_spider_mite": """
    Hama tungau dua bintik (Tetranychus urticae) menyerang daun tomat dengan mengisap cairan jaringan sehingga menyebabkan bercak-bercak kuning dan daun mengering.

    Pengendalian kimia dilakukan dengan akarisida yang mengandung abamectin, diaplikasikan sesuai petunjuk. Semprotan air secara berkala dapat membantu mengurangi populasi tungau.

    Penggunaan predator alami seperti kumbang ladybug efektif mengendalikan tungau tanpa merusak ekosistem. Hindari penggunaan pestisida yang membunuh predator alami untuk menjaga keseimbangan hayati.
""",

"Tomato___Target_Spot": """
    Target spot disebabkan oleh jamur Corynespora cassiicola yang menimbulkan bercak berwarna coklat dengan pola cincin menyerupai sasaran pada daun tomat.

    Fungisida berbahan aktif azoxystrobin efektif dalam pengendalian. Lakukan penyemprotan secara berkala terutama pada musim lembab dan basah.

    Rotasi tanaman, pengaturan jarak tanam agar tidak terlalu rapat, dan membuang daun yang terinfeksi membantu mencegah penyebaran penyakit ini.
""",

"Tomato___Tomato_Yellow_Leaf_Curl_Virus": """
    Virus Yellow Leaf Curl pada tomat ditularkan oleh serangga vektor whitefly (Bemisia tabaci). Gejala berupa daun menguning, menggulung ke atas, dan tanaman menjadi kerdil.

    Pengendalian dilakukan dengan mengontrol populasi whitefly menggunakan insektisida yang direkomendasikan dan perangkap serangga. Penggunaan varietas tomat tahan virus sangat dianjurkan.

    Buang dan musnahkan tanaman yang terinfeksi untuk mencegah penyebaran virus lebih luas.
""",

"Tomato___Tomato_mosaic_virus": """
    Tomato mosaic virus (ToMV) menyebabkan bercak mosaik kuning kehijauan pada daun, pertumbuhan terhambat, dan penurunan hasil panen. Virus ini dapat menyebar melalui peralatan berkebun dan kontak langsung.

    Gunakan varietas tahan virus ToMV dan lakukan sterilisasi peralatan berkebun secara rutin dengan desinfektan.

    Buang tanaman yang terinfeksi dan kontrol serangga vektor jika ada. Praktik sanitasi yang ketat dapat mencegah penyebaran virus.
""",
}

app = FastAPI(
    title="Plant Disease Classification API",
    description="API for classifying plant diseases using a Keras model with plant/leaf detection."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
input_size = DEFAULT_INPUT_SIZE

MAX_IMAGE_SIZE = 10 * 1024 * 1024

SUPPORTED_FORMATS = {'image/jpeg', 'image/png', 'image/jpg'}

class PlantDetector:
    """Service for detecting if an image contains plant leaves"""
    
    @staticmethod
    def analyze_color_features(img: Image.Image) -> Dict[str, float]:
        """
        Analyze color features to detect plant characteristics
        
        Args:
            img: PIL Image to analyze
            
        Returns:
            Dictionary with color feature scores
        """
        img_array = np.array(img)
        
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        lower_green1 = np.array([40, 40, 40])
        upper_green1 = np.array([80, 255, 255])
        
        lower_green2 = np.array([25, 40, 40])
        upper_green2 = np.array([40, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_green1, upper_green1)
        mask2 = cv2.inRange(hsv, lower_green2, upper_green2)
        green_mask = cv2.bitwise_or(mask1, mask2)
        
        total_pixels = img_array.shape[0] * img_array.shape[1]
        green_pixels = np.sum(green_mask > 0)
        green_percentage = green_pixels / total_pixels
        
        saturation = hsv[:, :, 1]
        avg_saturation = np.mean(saturation) / 255.0
        
        value = hsv[:, :, 2]
        avg_value = np.mean(value) / 255.0
        
        return {
            'green_percentage': float(green_percentage),
            'avg_saturation': float(avg_saturation),
            'avg_value': float(avg_value)
        }
    
    @staticmethod
    def analyze_texture_features(img: Image.Image) -> Dict[str, float]:
        """
        Analyze texture features that are common in plant leaves
        
        Args:
            img: PIL Image to analyze
            
        Returns:
            Dictionary with texture feature scores
        """
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        edge_density = np.mean(gradient_magnitude) / 255.0
        
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        texture_response = cv2.filter2D(gray, -1, kernel)
        texture_variance = np.var(texture_response) / (255.0 ** 2)
        
        return {
            'edge_density': float(edge_density),
            'texture_variance': float(texture_variance)
        }
    
    @staticmethod
    def detect_plant_leaf(img: Image.Image, threshold: float = 0.6) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Detect if the image contains a plant leaf based on multiple features
        
        Args:
            img: PIL Image to analyze
            threshold: Confidence threshold for plant detection
            
        Returns:
            Tuple of (is_plant_leaf, confidence_score, analysis_details)
        """
        try:
            analysis_size = (224, 224)
            img_resized = img.resize(analysis_size)
            
            color_features = PlantDetector.analyze_color_features(img_resized)
            
            texture_features = PlantDetector.analyze_texture_features(img_resized)
            
            green_score = min(color_features['green_percentage'] * 2.5, 1.0)
            
            saturation_score = color_features['avg_saturation']
            
            texture_score = min(texture_features['edge_density'] * 3.0, 1.0)
            
            brightness_score = min(color_features['avg_value'] * 1.5, 1.0)
            
            plant_confidence = (
                green_score * 0.4 +
                saturation_score * 0.25 +
                texture_score * 0.25 +
                brightness_score * 0.1
            )
            
            if color_features['avg_saturation'] < 0.1 and color_features['green_percentage'] < 0.05:
                plant_confidence *= 0.3
            
            if color_features['green_percentage'] < 0.02:
                plant_confidence *= 0.5
            
            is_plant = plant_confidence >= threshold
            
            analysis_details = {
                'color_features': color_features,
                'texture_features': texture_features,
                'scores': {
                    'green_score': green_score,
                    'saturation_score': saturation_score,
                    'texture_score': texture_score,
                    'brightness_score': brightness_score
                },
                'plant_confidence': plant_confidence,
                'threshold_used': threshold
            }
            
            logger.info(f"Plant detection - Confidence: {plant_confidence:.3f}, Is plant: {is_plant}")
            
            return is_plant, plant_confidence, analysis_details
            
        except Exception as e:
            logger.error(f"Error in plant detection: {str(e)}")
            return True, 0.5, {'error': str(e)}

class ModelService:
    """Service for model operations"""
    
    @staticmethod
    def load_model() -> Optional[tf.keras.Model]:
        """
        Load the Keras model from the specified path.
        
        Returns:
            Loaded Keras model or None if loading fails
        """
        global model, input_size
        
        try:
            if model is None:
                logger.info(f"Loading model from: {MODEL_PATH}")
                model = load_model(MODEL_PATH, compile=False)
                logger.info("Model loaded successfully")
                model.summary()
                
                input_size = ModelService.detect_input_size(model)
            return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    @staticmethod
    def detect_input_size(model: tf.keras.Model) -> Tuple[int, int]:
        """
        Detect the expected input size from the model.
        
        Args:
            model: Loaded Keras model
            
        Returns:
            Tuple of (height, width) for the model's expected input
        """
        try:
            input_shape = model.input_shape
            if input_shape and len(input_shape) == 4:
                height, width = input_shape[1], input_shape[2]
                if height is not None and width is not None:
                    logger.info(f"Detected model input size: ({height}, {width})")
                    return (height, width)
        except Exception as e:
            logger.warning(f"Could not detect model input size: {str(e)}")
        
        logger.info(f"Using default input size: {DEFAULT_INPUT_SIZE}")
        return DEFAULT_INPUT_SIZE

class ImageValidator:
    """Image validation service"""
    
    @staticmethod
    def validate_image_size(file_size: int) -> None:
        """
        Validate image file size
        
        Args:
            file_size: Size of the file in bytes
            
        Raises:
            ValueError: If file size exceeds the maximum allowed size
        """
        if file_size > MAX_IMAGE_SIZE:
            max_size_mb = MAX_IMAGE_SIZE / (1024 * 1024)
            raise ValueError(f"Ukuran gambar melebihi batas maksimum {max_size_mb}MB")
    
    @staticmethod
    def validate_mime_type(content_type: str) -> None:
        """
        Validate image MIME type
        
        Args:
            content_type: MIME type of the file
            
        Raises:
            ValueError: If file has an unsupported MIME type
        """
        if not content_type.startswith('image/'):
            raise ValueError("File yang diunggah bukan gambar")
        
        if content_type not in SUPPORTED_FORMATS:
            supported_formats = ', '.join(fmt.replace('image/', '') for fmt in SUPPORTED_FORMATS)
            raise ValueError(f"Format gambar tidak didukung. Format yang didukung: {supported_formats}")
    
    @staticmethod
    def validate_image_file(file_bytes: bytes) -> str:
        """
        Validate image file contents
        
        Args:
            file_bytes: Image file bytes
            
        Returns:
            Image format detected
            
        Raises:
            ValueError: If file is not a valid image
        """
        if not file_bytes:
            raise ValueError("File yang diunggah kosong")
        
        image_format = imghdr.what(None, file_bytes)
        if not image_format:
            raise ValueError("File yang diunggah bukan gambar yang valid")
        
        try:
            with Image.open(io.BytesIO(file_bytes)) as img:
                img.verify()
                return image_format
        except Exception as e:
            raise ValueError(f"File gambar tidak valid: {str(e)}")
    
    @staticmethod
    def validate_plant_content(img: Image.Image, strict_mode: bool = True) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate that the image contains plant/leaf content
        
        Args:
            img: PIL Image to validate
            strict_mode: Whether to use strict validation threshold
            
        Returns:
            Tuple of (is_valid, analysis_details)
            
        Raises:
            ValueError: If image doesn't contain plant content
        """
        threshold = 0.7 if strict_mode else 0.5
        is_plant, confidence, details = PlantDetector.detect_plant_leaf(img, threshold)
        
        if not is_plant:
            error_msg = (
                f"Gambar yang diunggah tampaknya tidak berisi daun tanaman. "
                f"Silakan unggah gambar yang menunjukkan daun tanaman untuk klasifikasi penyakit. "
            )
            raise ValueError(error_msg)
        
        return is_plant, details

class ImageProcessor:
    """Service for image processing operations"""
    
    @staticmethod
    def read_image(file_bytes: bytes) -> Image.Image:
        """
        Read image file bytes and convert to PIL Image.
        
        Args:
            file_bytes: Bytes of the image file
            
        Returns:
            PIL Image object in RGB format
        
        Raises:
            ValueError: If image cannot be read or processed
        """
        try:
            return Image.open(io.BytesIO(file_bytes)).convert("RGB")
        except UnidentifiedImageError as e:
            logger.error(f"Image format not recognized: {str(e)}")
            raise ValueError(f"Image format not recognized: {str(e)}")
        except Exception as e:
            logger.error(f"Error reading image: {str(e)}")
            raise ValueError(f"Error reading image: {str(e)}")
    
    @staticmethod
    def preprocess_image(img: Image.Image, target_size: Tuple[int, int]) -> List[np.ndarray]:
        """
        Preprocess image for model prediction using multiple methods.
        
        Args:
            img: PIL Image to preprocess
            target_size: Target size (height, width) for resizing
            
        Returns:
            List of preprocessed image arrays using different methods
        
        Raises:
            ValueError: If image cannot be preprocessed
        """
        try:
            img = img.resize(target_size)
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            
            normalized = img_array / 255.0
            mobilenet_preprocessed = preprocess_input(img_array.copy())
            
            return [normalized, mobilenet_preprocessed]
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise ValueError(f"Error preprocessing image: {str(e)}")

class PredictionService:
    """Service for prediction operations"""
    
    @staticmethod
    def format_class_name_english(class_name: str) -> str:
        """
        Format class name in English for reference.
        
        Args:
            class_name: Original class name with underscores
            
        Returns:
            Formatted class name in English
        """
        parts = class_name.split('___')
        if len(parts) == 2:
            plant, condition = parts
            plant = plant.replace('_', ' ')
            condition = condition.replace('_', ' ')
            return f"{plant} - {condition}"
        return class_name.replace('_', ' ')
    
    @staticmethod
    def format_class_name(class_name: str) -> str:
        """
        Format class name by translating to Indonesian and improving readability.
        
        Args:
            class_name: Original class name with underscores
            
        Returns:
            Formatted class name in Indonesian
        """
        if class_name in INDONESIAN_TRANSLATIONS:
            return INDONESIAN_TRANSLATIONS[class_name]
        
        parts = class_name.split('___')
        if len(parts) == 2:
            plant, condition = parts
            plant = plant.replace('_', ' ')
            condition = condition.replace('_', ' ')
            return f"{plant} - {condition}"
        return class_name.replace('_', ' ')
    
    @staticmethod
    def predict(img_array: np.ndarray, method_idx: int) -> Dict[str, Any]:
        """
        Make a prediction using the model.
        
        Args:
            img_array: Preprocessed image array
            method_idx: Index of preprocessing method used
            
        Returns:
            Dictionary with prediction results
            
        Raises:
            ValueError: If prediction fails
        """
        loaded_model = ModelService.load_model()
        if loaded_model is None:
            raise ValueError("Model failed to load")
        
        try:
            logger.info(f"Making prediction with method #{method_idx+1}, shape: {img_array.shape}")
            
            preds = loaded_model.predict(img_array)
            pred_idx = np.argmax(preds[0])
            
            raw_class = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else str(pred_idx)
            formatted_class = PredictionService.format_class_name(raw_class)
            
            confidence = float(np.max(preds[0]))
            confidence_percentage = f"{confidence * 100:.2f}%"
            
            suggestions = TREATMENT_SUGGESTIONS.get(raw_class, "Tidak ada saran pengobatan yang tersedia untuk penyakit ini.")
            
            logger.info(f"Prediction successful with method #{method_idx+1}")
            
            return {
                "prediction": formatted_class, 
                "raw_class": raw_class,
                "confidence": confidence_percentage,
                "preprocessing_method": method_idx+1,
                "prediction_english": PredictionService.format_class_name_english(raw_class),
                "suggestions": suggestions
            }
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise ValueError(f"Error making prediction: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Initialize model when application starts"""
    try:
        ModelService.load_model()
        logger.info("API started successfully")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    """Handle validation errors"""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=jsonable_encoder({"detail": exc.errors(), "message": "Validation error"}),
    )

@app.get("/")
def root() -> Dict[str, str]:
    """
    Health check endpoint.
    
    Returns:
        Dictionary with API status message
    """
    loaded_model = ModelService.load_model()
    if loaded_model is None:
        return {"message": "PERINGATAN: Model gagal dimuat. API tidak berfungsi penuh."}
    return {"message": "API Klasifikasi Penyakit Tanaman berjalan dengan deteksi tanaman yang diaktifkan."}

@app.post("/predict-disease")
async def predict(
    file: UploadFile = File(...),
    validate_image: bool = Query(True, description="Whether to perform comprehensive image validation"),
    validate_plant: bool = Query(True, description="Whether to validate that image contains plant leaves"),
    strict_plant_detection: bool = Query(True, description="Whether to use strict plant detection threshold")
) -> JSONResponse:
    """
    Predicts plant diseases from the uploaded image.

    Args:
        file: The uploaded image file.
        validate_image: Whether to perform comprehensive image validation.
        validate_plant: Whether to validate that the image contains plant leaves.
        strict_plant_detection: Whether to use a strict threshold for plant detection.

    Returns:
        JSONResponse with the prediction results.

    Raises:
        HTTPException: For various error conditions.
    """
    loaded_model = ModelService.load_model()
    if loaded_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="Model gagal dimuat. Silakan periksa log server."
        )
    
    try:
        ImageValidator.validate_mime_type(file.content_type)
        
        file_bytes = await file.read()
        
        ImageValidator.validate_image_size(len(file_bytes))
        
        if validate_image:
            image_format = ImageValidator.validate_image_file(file_bytes)
            logger.info(f"Image validated successfully, format: {image_format}")
        
        img = ImageProcessor.read_image(file_bytes)
        
        plant_analysis = None
        if validate_plant:
            is_plant, plant_analysis = ImageValidator.validate_plant_content(img, strict_plant_detection)
            logger.info(f"Plant validation passed with confidence: {plant_analysis.get('plant_confidence', 'N/A')}")
        
        preprocessed_images = ImageProcessor.preprocess_image(img, input_size)
        
        last_error = None
        for i, img_array in enumerate(preprocessed_images):
            try:
                prediction_result = PredictionService.predict(img_array, i)
                
                if plant_analysis:
                    plant_confidence = plant_analysis.get('plant_confidence', 0)
                    prediction_result['plant_detection'] = {
                        'confidence': f"{plant_confidence * 100:.2f}%",
                        'validated': True
                    }
                
                return JSONResponse(content=prediction_result)
            except Exception as e:
                last_error = str(e)
                logger.error(f"Preprocessing method #{i+1} failed: {last_error}")
                continue
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Semua metode preprocessing gagal. Error terakhir: {last_error}"
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Prediksi gagal: {str(e)}"
        )
