---
name: user-and-environment
description: Kullanıcı tercihleri (tam cümleli Türkçe yanıt), donanım, ortamlar, kardeş depolar ve voxcpm belleğine köprü
metadata:
  type: user
---

**Kullanıcı:** Serdar İ. Çağlar (ORCID 0000-0002-5776-2431; GitHub/HF
`serdarildercaglar` / HF dataset sahibi `serdarcaglar`). Türkçe konuşur.
**Yazım tercihi:** yanıtlar tam cümleli akıcı Türkçe olsun — telgraf dili,
tablo yığını ve aşırı madde işareti istenmiyor. **Why:** açık geri bildirim.
**How to apply:** raporları düzyazı ağırlıklı yaz; tabloyu yalnızca sayısal
karşılaştırmada kullan.

**Donanım/ortam:** RTX 3090 24 GB (kullan) + GTX 1650 (kullanma), 12 çekirdek,
62 GB RAM, 610 GB boş disk. Python: `/home/serdar/miniconda3/envs/main/bin/python`
(torch 2.5.1+cu124, transformers, snac 1.2.1 kurulu). Whisper servisi: docker
`whisper-vllm` (`docker start whisper-vllm`, OpenAI-uyumlu, port 8000).
LaTeX yerelde yok; gerekirse `texlive/texlive:latest` docker imajı.
Tarayıcı otomasyonu: kullanıcı `google-chrome --user-data-dir=$HOME/.chrome-cdp
--remote-debugging-port=9222` ile ayrı profil açar (Chrome 151 varsayılan
profille CDP'yi engelliyor), playwright `connect_over_cdp` ile bağlanılır.

**Kardeş depolar/veri:**
- Veri hattı: `/mnt/310C8DBF109E2BFC/projects/turkish-tts/voxcpm` →
  `github.com/serdarildercaglar/turkish-tts-audiobooks` (public). Eğitim verisi
  burada: `work/manifests/hf_train.jsonl` (416.315 klip), `hf_validation.jsonl`
  (8.502), sesler `work/clips` (yalnız train+validation; review sesi silinmiş),
  DNSMOS onnx `work/models/dnsmos_sig_bak_ovr.onnx`.
- Veri kümesi: HF `serdarcaglar/turkish-tts-audiobooks` (public, gated=manual;
  metrics/quality_metrics_full.parquet sidecar dahil). Şartlar: atıf + eğitilen
  modelin ticari-serbest **açık ağırlık** yayını zorunlu — bu repo Apache-2.0
  ile uyum sağlıyor.
- Makale: arXiv `submit/7981725` (EN, cs.CL+cs.SD) 24 Ağu 2026'da gönderildi,
  moderasyonda **on hold**; TR sürüm + eess.AS duyuruyu bekliyor. Ayrıntılar
  voxcpm projesinin belleğinde:
  `/home/serdar/.claude/projects/-mnt-310C8DBF109E2BFC-projects-turkish-tts-voxcpm/memory/`.
