---
name: ema-tts-baseline
description: "canberkkkkkk/ema-tts (NAR flow-matching Türkçe TTS) yerelde test edildi — RTF paritesi ve bizim stack'e alınacak dört kalem"
metadata: 
  node_type: memory
  type: project
  originSessionId: 294fa856-6770-4d4d-bf19-4f5d0446e990
  modified: 2026-08-27T14:30:39.983Z
---

27 Ağustos 2026'da `canberkkkkkk/ema-tts` (65M NAR flow-matching, VoxCPM2 AudioVAE
latent uzayı, tek ses, klonlama yok) yerelde kuruldu ve ölçüldü.
Kurulum: `/mnt/310C8DBF109E2BFC/projects/turkish-tts/ema-tts`, `main` conda ortamı;
tek bağımlılık `pip install --no-deps voxcpm` (audiovae alt ağacı yalnız torch/numpy/pydantic).

**Sonuç: mimari fikir aktarılamaz, dört somut kalem aktarılır.**

Çekirdek katkısı (sürekli Gauss hizalama, yuvarlanmış süre tekrarı yerine) NAR'a
özgü bir sorunun çözümü; AR'da süre kestiricisi yok, gradyan zaten her yere akıyor.

**RTF (DÜZELTİLMİŞ ölçüm).** İlk ölçümüm yanlış GPU'daydı: CUDA varsayılan
sıralaması (FASTEST_FIRST) 3090'ı `cuda:0` yapıyor, `nvidia-smi` PCI sırasıyla 1
gösteriyor — `CUDA_VISIBLE_DEVICES=1` GTX 1650'yi seçmiş. **Ölçümlerde daima
`CUDA_DEVICE_ORDER=PCI_BUS_ID` kullan.** Gerçek 3090'da ema-tts:
0,0196 (uzun) / 0,0412 (orta) / 0,120 (kısa IVR) / 0,248 (çok kısa);
DiT/ODE 258 ms + AudioVAE 12 ms. Üretim süresi uzunluktan neredeyse bağımsız
(~0,26–0,33 s sabit). Yani NAR uzun metinde bizim AR'ımızdan (0,09–0,10) ~5 kat
hızlı ve README'nin "RTF 0,035" iddiası tutarlı. Bizim kalan avantajımız kısa
IVR anonsunda akış: onların sabit ~0,26 s gecikmesine karşı ilk paket ~50–80 ms.
NAR'a geçmek yine de gerekmiyor, ama "parite" gerekçesi geçersiz.

**Alınacaklar:** (a) 8 kHz banda indirilmiş eval — src/evaluate.py hiç banda
indirmiyor, hattan teslim edilmeyecek kaliteyi ölçüyoruz; (b) metin ön işleme
(text.py, Apache-2.0) — bizde hiç yok, ekin okunuşa göre yeniden hesaplanması
iyi iş, 5 hatası düzeltilmeli (alfanümerik içi rakam genişlemiyor, eksi işareti
kayboluyor, km/sa bölünüyor, IBAN→ıban, @ ve # UNK); (c) klip başlangıcı — ilk
kelime CER'i pilotta ayrı ölçülmeli; (d) 3 seed ortalaması ± ile raporlama.

**Prozodide üstünlük bizde:** ema-tts'in prozodi kontrolü yok. Bizde klon kipi
bedava prozodi aktarımı veriyor; 55 boş vokab yuvasına konuşma hızı/F0/enerji
kova tokenları eklenebilir (pilot öncesi karar — paketlenmiş veriyi değiştirir).

**Why:** kendi mimarimizi çatmadan önce dışarıdaki en iyi açık Türkçe sistemi
taban çizgisi olarak ölçmek istendi; 65M ile %3,0 WER alınıyorsa darboğaz
parametre sayısı değil metin→akustik eşlemesi, bu da derinlik başı kararında
harcanacak kalite payı olduğunu gösteriyor.

**How to apply:** RTF tartışması yeniden açılırsa NAR seçeneği ölçülmüş olarak
elenmiş sayılır. Frontend ve 8 kHz eval pilot öncesi iş kalemi.
İlgili: [[eval-digit-scoring-trap]]
