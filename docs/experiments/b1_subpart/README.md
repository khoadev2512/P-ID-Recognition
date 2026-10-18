# B1: per-subpart augment, bằng chứng chọn thiết kế (2026-10-18)

Bối cảnh và kết luận: `docs/PLAN_STAGE2_ONESHOT.md` §B1. Code chính thức:
`src/utils/augment.py` (`SubpartWarp`).

## Ảnh

| File | Nội dung |
|---|---|
| `B1_review.png` | **Ảnh cần duyệt.** Crop thật (viền xanh) + 4 bản `SubpartWarp` từ code chính thức: S1/3/7/10/13/14 |
| `B1_review_pairs.png` | **Ảnh cần duyệt.** Cặp dễ nhầm S12↔S13, S15↔S16, S2↔S5: chi tiết phân biệt phải còn |
| `valves_1_8.png`, `valves_9_16.png` | Crop gốc của 16 class valve (margin 10% như `pid-prep-crops`) |
| `iso_valves.png` | Template ISO valve trong `data/raw/iso_symbols_clean/` (X8071 bị lỗi) |
| `proto_cc.png` | Phương án 1: affine theo connected component (thiết kế ban đầu) → valve chỉ bị xoay cả khối |
| `proto_cluster.png` | Chia nét bằng k-means → nét rách (bỏ) |
| `proto_warp.png` | Phương án 3 bản thử nghiệm (warp 6%; bản chính thức dùng 4%) |

## Số liệu

`results.jsonl`: mỗi dòng là một lần chạy mini-P1. `mode` có các giá trị:
- `none`: không augment theo bộ phận
- `cc`: phương án 1
- `warp`: phương án 3, bản thử nghiệm
- `real`: `SubpartWarp` chính thức

Lần chạy có `epochs: 0` là baseline chỉ dùng ImageNet. Các trường:
- `iso_acc`: one-shot từ template ISO
- `crop_acc`: one-shot từ 1 crop DigitizePID
- `seen_val_acc`: độ chính xác trên 12 class đã train

## Chạy lại

Code thử nghiệm, không thuộc pipeline. Chạy từ gốc repo; `$D` = thư mục này, `$C` = file cache tạm.

```bash
D=docs/experiments/b1_subpart; C=/tmp/valve_crops.pkl
uv run python $D/code/extract.py data/canonical $C        # cache crop valve (~41MB, ~2 phút)
uv run python $D/code/cc_stats.py data/canonical          # tỉ lệ crop tách được >=2 component
uv run python $D/code/stroke.py $C $D/code data/raw/iso_symbols_clean   # độ dày nét ISO vs crop
uv run python $D/code/minip1.py $C $D/code data/raw/iso_symbols_clean real 0 6   # 1 lần ~1 phút (MPS)
uv run python $D/code/review.py $C $D/B1_review.png Symbol_1,Symbol_3,Symbol_7,Symbol_10,Symbol_13,Symbol_14
```

`augproto.py` / `proto.py` là bản thử nghiệm của các phương án (modes `cc`, `cluster`, `warp`).
