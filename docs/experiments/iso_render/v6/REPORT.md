# V6 — Sinh nhiều biến thể + lọc trùng (báo cáo, 2026-11-05)

Commit: xem `git log` (`feat: RENDER_ISO V6 variants and dedup`). 533 test (516 cũ + 17 mới), ruff sạch trên file đã đụng.

## 1. Đã làm gì

| File | Vai trò |
|---|---|
| `src/utils/diversity.py` | aHash / dHash tự viết (PIL + numpy, ô lưới chia theo mean-box, đảo để giá trị = độ phủ mực), `hamming`, `Deduper` |
| `src/utils/iso_variants.py` | `derive_seed`, `variant_settings` (đọc + kiểm `variants:`), `plan_variant` (V3 + stroke/size), `render_variant` (V4 + V5), `assignment_signature`, `quality_failures`, `generate_sheet` (vòng lặp lọc + ghi file) |
| `configs/render_iso.yaml` | khối mới `render_iso.variants` (không đụng `configs/config.yaml`) |
| `scripts/make_variants.py` | chạy cả 12 tờ song song (1 process/tờ), ghi `data/iso_open100/variants/` + CSV thống kê |
| `scripts/check_variant_style.py` | chạy G1 trên lưới (w, hệ số cỡ) với cùng phép gán + bố cục |
| `scripts/make_v6_figures.py` | vẽ ảnh bằng chứng |
| `scripts/check_g1.py` | tách `evaluate(page, runs, source, cfg)` khỏi `main` (dùng lại cho bộ lọc); **đổi cách đo tiêu chí 2** (xem 3.3) |
| `tests/test_iso_variants.py` | 17 test |

Không sửa `assign_symbols`, `transform_layout`, `render_page` (mọi khác biệt đi qua tham số): `w` qua bản sao `cfg` với `page.stroke_px` đổi; cỡ ký hiệu qua bản sao `source` có hộp ký hiệu (`sized_nodes`) nhân hệ số quanh tâm (V5 vừa mỗi ký hiệu vào hộp nguồn). Không đụng Stage 1/2, `pyproject.toml`, `third_party/`.

## 2. Quyết định thiết kế

1. **RNG độc lập theo giai đoạn.** `derive_seed(base, sheet, index, stage)` = blake2b của chuỗi `base|sheet|index|stage` (ổn định giữa các lần chạy, không dùng `hash()` của Python). Ba giai đoạn `assign` / `layout` / `style` mỗi cái một `random.Random`. Biến thể `i` của tờ `s` luôn như nhau dù sinh trước hay sau biến thể khác. Test ghim giá trị seed (đổi cách dẫn xuất sẽ làm seed lưu trong JSON mất nghĩa). Ảnh tái dựng được từ (sheet, index, base_seed): test so SVG sinh lại với SVG đã ghi.
2. **Nguồn biến thể**: phép gán V3; bố cục V4 (`gap_scale` + ánh xạ trục); `w` ~ U[2, 4] px (= [2, 4]·S/3, làm tròn 0.01, liên tục chứ không chỉ 2/3/4); hệ số cỡ ~ U[0.85, 1.15] **một hệ số cho cả tờ** (không phải từng ký hiệu).
3. **Hệ số cỡ chỉ áp lên `sized_nodes`** = ký hiệu ISO đặt theo cổng + mũi tên dòng chảy. Bình `multi`, hộp, bong bóng, cờ vẽ trên hộp V4 nên không nhân. Lần đầu nhân hết thì tiêu chí 3 trượt 18/36 ở hệ số 0.85 (hộp bình nguồn co lại nên mũi tên trong bbox bình không còn "chạm" bình trong nguồn, trong khi bình vẫn vẽ nguyên) — lỗi của cách đo, đã sửa bằng `sized_nodes`.
4. **Thứ tự lọc** (rẻ trước): chữ ký phép gán trùng (trước khi vẽ) → lỗi V5 (bắt exception, ghi `error`) → `Page.log` vượt ngưỡng (`quality.max_log`: `bubbles_unsettled`, `unmatched_ports` đều tối đa 0) → aHash+dHash trùng → tiêu chí G1. `loose_ends`, `agitators_skipped`, `shrunk` **không** lọc: thấy ở cả bản gốc đạt G1 (vd. sheet 3 seed 0 `loose_ends=4`) và là hành vi thiết kế.
5. **Trùng ảnh** = **cả hai** hash cách một ảnh đã giữ < `min_distance` = **8 bit** (/256, lưới 16×16). Lý do chọn 8: khoảng cách cùng tờ không có hai cụm tách bạch (`hash_calibration.txt`: percentile 5 = 12, trung vị láng giềng gần nhất 13, min 3); xem cặp ở khoảng cách 5 / 8 / 11 thì bố cục gần như giống nhau, chỉ khác ký hiệu. 12 làm tờ 1 mất 47/60 ứng viên, 8 mất 31/60. **Đây là chọn theo cảm quan, cần user duyệt** (mục 6).
6. **Chữ ký phép gán** chỉ gồm lựa chọn V3 (nút, loại, template + xoay/lật, flanges, agitator, clearance, hướng mũi tên). Chạy thật: 0 trùng chữ ký (cả 552 lần thử), nên test chỉ kiểm hàm; nhánh `signature` trong `generate_sheet` không có test riêng với dữ liệu thật (chỉ kiểm hàm chữ ký đổi theo từng lựa chọn).
7. **Chỉ tiêu G1 làm bộ lọc cuối** (cấu hình `g1.drop` / `g1.flag`): mặc định **drop** cả 1, 2, 3, 5, 6, 7. **Quyết định cho tiêu chí 5 (mũi tên–`general` tờ 10 và họ hàng): LOẠI, không giữ-và-gắn-cờ.** Lý do: hở 2–12 px giữa mũi tên và ký hiệu mà đồ thị nói là nối nhau là lỗi nhãn-thật; giữ thì phải cho chạy kèm cờ ở mọi chỗ dùng. Cái giá: tờ 10 chỉ còn 32/36 sau 60 lần thử (28 bị loại, đều tiêu chí 5 trừ 1), tờ 4 còn 31 (20 loại), và lỗi này **không chỉ ở tờ 10** (xem 3.2). Đổi sang gắn cờ chỉ cần `g1: {drop: [1,2,3,6,7], flag: [5]}` (đã có test cho cả hai nhánh); JSON ghi `g1_flags`.

## 3. Kết quả

### 3.1 Sinh đầy đủ (`make_variants.log`, `per_sheet.csv`, `attempts.csv`)

Lệnh: `.venv/bin/python scripts/make_variants.py --out data/iso_open100/variants --stats docs/experiments/iso_render/v6 --workers 10`

| Tờ | thử | giữ | trùng ảnh | G1 loại | chữ ký / lỗi / chất lượng |
|---|---|---|---|---|---|
| 0 | 41 | 36 | 2 | 3 | 0 |
| 1 | 60 | 26 | 31 | 3 | 0 |
| 2 | 47 | 36 | 1 | 10 | 0 |
| 3 | 38 | 36 | 2 | 0 | 0 |
| 4 | 60 | 31 | 9 | 20 | 0 |
| 5 | 38 | 36 | 2 | 0 | 0 |
| 6 | 48 | 36 | 5 | 7 | 0 |
| 7 | 40 | 36 | 3 | 1 | 0 |
| 8 | 39 | 36 | 3 | 0 | 0 |
| 9 | 36 | 36 | 0 | 0 | 0 |
| 10 | 60 | 32 | 0 | 28 | 0 |
| 11 | 45 | 36 | 8 | 1 | 0 |
| **Tổng** | **552** | **413** | **66** | **73** | **0** |

- G1 loại 73 = 60 (tiêu chí 5) + 13 (tiêu chí 1). Tiêu chí 5: tờ 10 (27), tờ 4 (20), tờ 2 (10), tờ 1 (3). Tiêu chí 1: tờ 6 (7), tờ 0 (3), tờ 7 và 10 (1).
- Bộ lọc `quality` (bubbles_unsettled / unmatched_ports) loại **0** ứng viên với base_seed 0; mọi biến thể giữ có 2 counter này = 0. Nó là bảo hiểm cho seed khác, chưa chứng minh được tác dụng trên dữ liệu thật.
- Tờ 1 có nhiều bản trùng nhất (31/60): ít cột/hàng nên bố cục mọi biến thể gần nhau; vì `max_attempts = 60` nên chỉ giữ 26.
- **Khoảng cách Hamming** (`hamming_hist.png`, `hash_calibration.txt`): các ứng viên bị loại có khoảng cách tới ảnh giữ gần nhất 3 (2), 4 (7), 5 (11), 6 (18), 7 (28); các ứng viên còn lại: min 8, p10 9, p25 11, trung vị 14, p75 19, max 43. Không có khe tách: ngưỡng cắt giữa một liên tục.
- **Thời gian / dung lượng** (10 tiến trình, máy 10 lõi): 552 lần thử trong 52 s tường (244 s CPU). Mỗi biến thể giữ: 0.70 s (gán 0.06, vẽ+raster 0.11, hash 0.02, G1 0.33, ghi 0.18); lần thử bị loại trung bình 0.31 s. Đĩa: 413 biến thể = 127 MB, **~308 KB/biến thể** (PNG xám 207 KB, SVG 81 KB, JSON 29 KB); ảnh ~7000×4600 px. Một lần 36 × 12 = 432 biến thể chỉ ~130 MB. V7 (suy giảm trên ảnh 30 MP) mới là chi phí lớn.

### 3.2 G1 theo `w` và cỡ ký hiệu (`g1_style.csv`, `g1_style.log`)

Cùng phép gán + bố cục (3 chỉ số mỗi tờ × 12 tờ = 36 lượt mỗi ô), chỉ đổi `w` và hệ số cỡ. Số lượt **trượt** (trên 36), cột `đạt` = đạt cả 6 tiêu chí:

| w | cỡ | TC1 | TC2 | TC3 | TC5 | TC6 | TC7 | đạt |
|---|---|---|---|---|---|---|---|---|
| 2.0 | 0.85 | 1 | 0 | 0 | 8 | 0 | 0 | 27/36 |
| 2.0 | 1.0 | 3 | 0 | 0 | 3 | 0 | 0 | 30/36 |
| 2.0 | 1.15 | 8 | 0 | 0 | 1 | 0 | 0 | 27/36 |
| 2.5 | 0.85 | 0 | 0 | 0 | 8 | 0 | 0 | 28/36 |
| 2.5 | 1.0 | 1 | 0 | 0 | 3 | 0 | 0 | 32/36 |
| 2.5 | 1.15 | 2 | 0 | 0 | 1 | 0 | 0 | 33/36 |
| 3.0 | 0.85 | 0 | 0 | 0 | 6 | 0 | 0 | 30/36 |
| 3.0 | 1.0 | 0 | 0 | 0 | 1 | 0 | 0 | 35/36 |
| 3.0 | 1.15 | 2 | 0 | 0 | 1 | 0 | 0 | 33/36 |
| 3.5 | 0.85 | 0 | 0 | 0 | 6 | 0 | 0 | 30/36 |
| 3.5 | 1.0 | 0 | 0 | 0 | 1 | 0 | 0 | 35/36 |
| 3.5 | 1.15 | 0 | 0 | 0 | 1 | 0 | 0 | 35/36 |
| 4.0 | 0.85 | 1 | 0 | 0 | 5 | 0 | 0 | 30/36 |
| 4.0 | 1.0 | 1 | 0 | 0 | 1 | 0 | 0 | 34/36 |
| 4.0 | 1.15 | 1 | 0 | 0 | 1 | 0 | 0 | 34/36 |

Đọc kết quả:
- **Tiêu chí 2, 3, 6, 7 đạt hết** ở mọi w và cỡ (sau hai sửa ở 2.3 và 3.3). Không đổi ngưỡng nào của `check_g1`.
- **Tiêu chí 1** trượt nhiều hơn ở w = 2 và cỡ lớn (8/36 ở 2.0/1.15). Tất cả 8 ca xem được đều là cổng **`loose`** của van có thân chéo (`safety_valve_spring`, `check_valve_general`) mà ống chéo dừng cách mực ký hiệu ~1–1.5 px; nét 2 px mỏng nên khe này hiện ra (cùng loại lỗi `valve60` đã biết ở V5b). Với w = 4, `general101` (tờ 6) ống bắt đầu cách cổng 1.7 px trên cùng hàng với nét ký hiệu — không thấy hở bằng mắt. Không nới ngưỡng 1 px; những ca này bị bộ lọc loại (13 ca / 552).
- **Tiêu chí 5** trượt nhiều khi cỡ 0.85 (5–8 so với 1–3): ký hiệu co nhỏ rời khỏi mũi tên/ký hiệu sát bên không có ống nối (`arrow59–pump75` 6.1 px, `arrow129–valve81` 12.8 px ở cỡ 0.85). Cùng họ với lỗi tờ 10, **không phải lỗi mới do biến thể, nhưng cỡ nhỏ làm nó hiện nhiều hơn**. Do loại các ca này, bộ dữ liệu hơi lệch khỏi cỡ nhỏ ở tờ 1, 2, 4, 10 (chưa đo mức lệch).
- Ở w = 3, cỡ 1.0 (cấu hình cũ): 35/36, ca trượt là tờ 2 (không phải tờ 10): `arrow129–valve81` hở 11.1 px.

### 3.3 Hai chỗ `check_g1.py` phải sửa

- Tách `evaluate()` khỏi `main()` (logic y nguyên).
- **Tiêu chí 2 (độ dày nét)**: bộ đo cũ lấy trung vị khoảng cách-nét ở độ phân giải 1×, nên là số nguyên: nét chẵn đọc thiếu 1 px (2 px → 1) và nét lẻ phân số hay lệch (w = 2.34: ký hiệu đọc 1.0, ống 3.0) → báo trượt dù cả hai vẽ cùng `stroke-width`. Với w liên tục của V6 thì tiêu chí này trượt giả ở nhiều biến thể. Bộ đo mới đo ở 4× trên 8 ô 200 px nhiều mực nhất của từng lớp: sai số ≤ 0.25 px (kiểm: w = 2, 2.5, 3, 3.7, 4 đọc 1.75/2.25/2.75/3.25–3.75/3.75, hai lớp lệch nhau ≤ 0.5). Ngưỡng 1 px giữ nguyên. Test `test_stroke_thickness_reads_fractional_and_even_widths`. **Kết quả G1 cũ của V5/V5b khi ghi "symbols 3.00 px" nay đọc 2.75**; không ảnh hưởng đạt/trượt.

### 3.4 Độ phủ lớp (`per_class.csv`, `per_fine_name.csv`; 413 biến thể giữ)

COCO `label` (số thực thể; số biến thể có lớp đó):

| Lớp | thực thể | biến thể | ghi chú |
|---|---|---|---|
| `agitator_general` / `_impeller` / `_turbine` | 55 / 60 / 65 | 50 / 54 / 56 | tổng 180 agitator trong 124 biến thể (30 biến thể có agitator bị bỏ: `agitators_skipped`); dồn ở tờ 9, 10, 11 (tờ 9: 15 + 18 + 22), rồi tờ 2 (7 + 15 + 16) |
| `column` | 108 | 108 | chỉ có 3 nút cột (tờ 0, 5, 6) → mỗi tờ 36 biến thể × 1 cột; **chỉ 3 nút khác nhau**, chỉ khác phong cách vẽ/bố cục, không khác topology |
| `flanged_connection` | 3964 | 390 | tờ 8 chỉ 14 |
| G5 trong vocab: `breather_valve`, `silencer`, `viewing_glass` | 181, 192, 164 | 140, 152, 127 | |
| G5 ngoài vocab (nhãn COCO `other_iso`, tên chi tiết trong `per_fine_name.csv`) | flame_arrestor (3 loại) 559, funnel 196, strainer(+cone) 336, steam_trap 159, rupture_disc 178 | 109–157 | |
| đĩa: `blind_disc`, `interchangeable_disc`, `_open` | 617, 613, 617 | 281, 277, 260 | `orifice_plate` (ngoài vocab) 162 |
| bơm: `pump_centrifugal` / `gear` / `general` | 145 / 127 / 127 | 93 / 80 / 88 | chỉ tờ 1, 3, 5, 7 |
| `valve_three_way` | 40 | 40 | chỉ tờ 4, 8, 10 (3 nút) |
| `other_isa`, `other_generic`, `other_iso` | 10739, 1544, 4844 | | `other_iso` gồm 1496 bình 2062 |

Lớp có ít **nút nguồn** (cột 3, bơm 16, bình 35, `valve_three_way`) không được 36 biến thể làm đa dạng thật: chỉ bù phong cách/vị trí. Agitator: p = 0.3 trên ~20 bình đủ điều kiện nên biến thể có 0 đến 5 agitator; chỉ 124/413 (30%) biến thể có ít nhất một, 289 biến thể không có.

## 4. Ảnh bằng chứng (cục bộ; `*.png` trong `docs/experiments/` bị gitignore)

- `docs/experiments/iso_render/v6/montage_sheet3.png` — 6 biến thể giữ của tờ 3 (đã xem: bố cục cùng topology, ký hiệu và nét khác nhau, w ghi ở góc).
- `docs/experiments/iso_render/v6/duplicate_pair_0_sheet0.png`, `duplicate_pair_1_sheet7.png` — ảnh giữ cạnh ảnh bị loại và khoảng cách (cặp tờ 7: 6 < 8; đã xem: hai ảnh cùng bố cục, chỉ khác vài ký hiệu — nên đây đúng là chỗ hash loại cả biến thể "khác ký hiệu", xem hạn chế).
- `docs/experiments/iso_render/v6/style_crops.png` — cùng ký hiệu ở w = 2 / 3 / 4 và cỡ 0.85 / 1.15 (đã xem: nét dày lên đồng đều cả ống lẫn ký hiệu, cỡ đổi mà ống vẫn nối).
- `docs/experiments/iso_render/v6/hamming_hist.png` — phân bố khoảng cách tới ảnh gần nhất đã giữ, kèm ngưỡng.
- Dữ liệu: `attempts.csv` (từng lần thử: lý do, khoảng cách, giây), `per_sheet.csv`, `per_class.csv`, `per_fine_name.csv`, `g1_style.csv`, `g1_style.log`, `make_variants.log`, `nodedup_per_sheet.csv` + `hash_calibration.txt` (240 biến thể không lọc trùng để chọn ngưỡng).

Sản phẩm sinh ra (gitignore, `data/*`): `data/iso_open100/variants/sheet<i>/v<NNN>.{png,svg,json}` và `manifest.csv`. JSON: seeds từng giai đoạn, `stroke_px`, `size_factor`, `signature`, `ahash`/`dhash` (hex), `nearest`, `log`, `g1`, `g1_flags`, `annotations` (bbox mực, `added` cho agitator). Chưa có GraphML đổi nhãn / COCO: là việc V8 (dựng lại từ seeds nếu cần).

## 5. Hạn chế

- Hash 16×16 chỉ thấy bố cục thô: hai biến thể cùng bố cục nhưng khác phép gán ký hiệu bị coi là trùng, dù đó chính là nguồn đa dạng. Chấp nhận vì ngưỡng thấp (8 bit) chỉ loại 12% lần thử; nếu muốn giữ các biến thể "khác ký hiệu" thì đổi thành hash lưới 32×32 hoặc bỏ lọc ảnh và chỉ dùng chữ ký.
- Mỗi tờ nguồn chỉ 1 topology: 36 biến thể không thay được thêm bản vẽ gốc; chia tập V8 phải theo tờ gốc.
- Hệ số cỡ chung cả tờ; không có biến thiên cỡ từng ký hiệu.
- Số biến thể không đều (26–36/tờ) do `max_attempts = 60`; tờ 1 thiếu vì bố cục ít biến đổi, tờ 4/10 vì tiêu chí 5.
- Biến thể loại bởi TC5/TC1 làm lệch nhẹ phân bố (ít cỡ nhỏ, ít `safety_valve_spring`/`check_valve_general` có cổng loose ở w = 2); chưa đo mức lệch.
- `quality.max_log` chưa từng kích hoạt (0/552).
- Không đo trên máy khác; thời gian là máy 10 lõi macOS.

## 6. Câu hỏi cho user

1. `min_distance = 8` (trên 256 bit, cả hai hash): ổn? Hay muốn lọc yếu hơn (chỉ chữ ký) / hash 32×32 để không loại bản "cùng bố cục, khác ký hiệu"?
2. Tiêu chí 5: **loại** (hiện tại) hay **giữ + gắn cờ**? Loại ⇒ tờ 4, 10 chỉ ~31–32 biến thể và lệch khỏi cỡ nhỏ.
3. 36 biến thể/tờ (~430 ảnh, ~130 MB) hay số khác? `max_attempts` 60 có đủ?
4. `w` liên tục U[2, 4] hay chỉ {2, 3, 4}? (V7 giới hạn suy giảm theo `w`, giá trị liên tục thì vẫn dùng được.)
5. Chấp nhận đổi cách đo tiêu chí 2 của `check_g1.py` (4×, 8 ô)?
