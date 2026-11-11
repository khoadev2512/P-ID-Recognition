# V7 — Làm suy giảm toàn ảnh (báo cáo, 2026-11-06)

Commit: xem `git log` (`feat: RENDER_ISO V7 degradation`). 548 test (533 cũ + 15 mới), ruff sạch trên file đã đụng.
Các quyết định V6 (min_distance 8, loại tiêu chí 5, đo lại tiêu chí 2 của `check_g1`) giữ nguyên.

> **Chạy lại 2026-11-06** sau khi V6 đổi sang lọc trùng theo nội dung (V6 REPORT mục 0): cùng lệnh ở mục 5,
> **428 ảnh**, 92 s, 4.2 MB/ảnh (`make_degraded.txt`, `degraded.csv` đã ghi đè). 387 ảnh có ở cả hai lần trùng
> từng byte (JPEG), vì seed `degrade` và phép suy giảm không đổi. Các số đo bên dưới (survival, geometry,
> tune) không phụ thuộc tập biến thể nên không chạy lại; số "413" bên dưới là của lần chạy đầu.
> **Chạy lại lần 2** (cùng ngày) sau khi V5 nối khe giữa hai ký hiệu nối thẳng (V6 REPORT mục 0b):
> **432 ảnh**, 102 s, 4.1 MB/ảnh; `make_degraded.txt`, `degraded.csv` đã ghi đè.
> **Chạy lại lần 3** (2026-11-08) sau V5c (trang có chữ): 432 ảnh, 232 s. Chữ còn đọc được ở mòn mạnh nhất
> (nhạt hơn ký hiệu vì nét chữ mảnh hơn). JSON suy giảm chưa mang `texts` (chữ không phải nhãn COCO).

## 1. Đã làm gì

| File | Vai trò |
|---|---|
| `src/utils/iso_degrade.py` | `degrade_image` (morph, nghiêng/xoay, pixelate, blur, ánh sáng, nhiễu, JPEG), tham số theo `w` (`limits`, `clamp_params`, `peak_contrast`), affine (`affine_for`, `apply_affine`), hộp theo mực (`ink_box_after`), `moved_geometry` |
| `configs/render_iso.yaml` | khối mới `render_iso.degrade` |
| `src/utils/iso_variants.py` | thêm giai đoạn `degrade` vào `STAGES` (seed riêng); `Rendered.sheet` (sheet sau V4). Không đổi giá trị seed/ảnh của V3/V4/V5 |
| `scripts/make_degraded.py` | làm suy giảm cả bộ V6, ghi `data/iso_open100/degraded/` |
| `scripts/check_degrade_survival.py` | kiểm "chi tiết còn sống" (NCC; vùng khác biệt cho cặp NCC không phân biệt được) + dò tham số tối đa (`--tune`) |
| `scripts/check_degrade_geometry.py` | so hộp đã dời với hộp đo trên ảnh đã xoay |
| `scripts/make_v7_figures.py` | ảnh bằng chứng |
| `tests/test_iso_degrade.py` | 15 test |

`degrade` là bản **viết lại** (không import) ý tưởng noise/blur/pixelate của `scripts/render_pid.py:degrade`; thêm dilation/erosion, xoay + nghiêng, ánh sáng không đều, JPEG.

## 2. Thiết kế

**Thứ tự** (như một lần quét): morph (nét dày/mỏng) → xoay/nghiêng (`cv2.warpAffine`, nền trắng, giữ kích thước trang) → pixelate → blur → ánh sáng (hệ số mượt 4×6 phóng to, nhân) → nhiễu Gauss (theo dải 512 hàng, bộ nhớ nhỏ) → JPEG. Ảnh lưu chính là file JPEG đó.

**Giới hạn theo `w`** (w = `stroke_px` của biến thể, liên tục trong [2, 4]; `config` ghi tỉ lệ, `clamp_params` ép):

| Tham số | Giới hạn | Cách hiểu |
|---|---|---|
| `morph_px` (đổi bề rộng nét; < 0 mòn, > 0 nở) | \|morph\| ≤ w/2 | **bề rộng nét đổi tối đa w/2** (nhân tử kernel 2r+1 với r = \|morph\|/2; r lẻ thì trộn hai kernel kề nhau) |
| `blur_sigma` | ≤ w/2 px | đúng theo plan |
| `pixelate_block` | ≤ max(1, w/2) | w = 2 thì không pixelate |
| **ràng buộc thêm** `peak_contrast` | ≥ 0.55 | xem dưới |

Ràng buộc thêm vì từng giới hạn riêng chưa đủ: mòn w/2 **và** blur w/2 cùng lúc khiến nét w = 2 chỉ còn ~40% độ đậm (test đầu tiên bắt được: không còn pixel < 128). `peak_contrast(w, p)` = độ đậm giữa một nét thẳng bề rộng w sau morph + pixelate + blur, **đo trên nét vẽ thật** (4 pha dưới pixel, lấy pha mờ nhất; không tính hình học/nhiễu). `clamp_params` hạ blur/pixelate rồi hạ mòn cho tới khi ≥ `MIN_PEAK` = 0.55. Kết quả đo (`thin_line.txt`, w = 2…4, tất cả ở mức mạnh nhất, mòn):

| w | độ đậm giữa nét | độ đậm còn lại trên cả trang (sheet 3) |
|---|---|---|
| 2.0 | 0.58 | 0.68 |
| 3.0 | 0.57 | 0.52 |
| 4.0 | 0.59 | 0.51 |

Tức nét mỏng nhất vẫn xám đậm hơn nửa ink, không biến mất; nở mạnh nhất: 0.82–0.99.

**Khoảng tham số** (`degrade:` trong config; mỗi tham số rút đều trong khoảng, 1 tham số/lần, thứ tự cố định):
xoay [−1, 1]°, nghiêng ngang [−0.5, 0.5]°, morph [−1, **0.4**]·w/2, blur [0, **0.35**]·w/2, pixelate [0, **0.5**], ánh sáng [0, 0.2], nhiễu σ [0, 10], JPEG q [55, 90]. Các số **in đậm** là kết quả hạ trần theo mục 3.

**RNG riêng**: `derive_seed(base, tờ, chỉ số, "degrade")`. Thêm "degrade" vào `STAGES` chỉ thêm một khoá vào `seeds` của biến thể **mới**; seed `assign`/`layout`/`style` và chữ ký phép gán không đổi (test so với JSON V6 trên đĩa: cùng `signature` và 3 seed). Trang vẽ lại từ seed phải **trùng từng pixel** với PNG sạch (`degrade_one` báo lỗi nếu không): 413/413 trùng.

## 3. Kiểm "chi tiết còn sống" (`survival.csv`, `survival.txt`, `tune.txt`, `survival_before_tuning.txt`)

Mỗi template vẽ riêng (k px/mm × w px, đặt giữa trang trắng) → suy giảm ở **mức tối đa** → NCC (`TM_CCOEFF_NORMED`, trượt ±4 px) với bản sạch của mọi template cùng nhóm; **đạt** khi khớp nhất là template cùng lớp (`fine_name`). Nhóm: G1 (13), G2 (7), G5 (8), 3 agitator (so với nhau), `flanged_connection` (so với cả G3, 11). 96 template-lớp × (k ∈ {3.0, 3.5, 5.5, 7.5}) × (w ∈ {2, 3, 4}) × 12 kịch bản = 4608 phép đo. Kịch bản: mọi tham số mạnh nhất cùng lúc (mòn / nở) và từng tham số riêng ở mức mạnh nhất.

**k được gác (gate)**: k ≥ 3.5 px/mm. Đo từ 4 biến thể × 12 tờ: k của van G1 có p5 = 3.2, trung vị 6.3; G5 2.9–4.5; 511 p5 = 5.1. k = 3.0 chỉ báo cáo.

**Trần đã hạ** (trước khi hạ, k ≥ 3.5: tất cả tham số ở mức tối đa của plan, không loại mẫu nào):
- mòn mạnh nhất: đạt hết.
- **nở (morph > 0)**: một mình chịu tới 0.8·w/2, cộng với blur/pixelate thì **còn 0.4·w/2**: van chắn `valve_gate` X8074 bị nhận thành `valve_butterfly` X8075_1 (cả ở k = 7.5, w ≥ 3), `valve_general` 2101 → `valve_globe`, `valve_needle` X8076 → `valve_globe`, `check_valve_general` X8077 → `check_valve_globe`, `valve_three_way` 2103 → `valve_three_way_globe`.
- **blur 1.0 → 0.35** (một mình chịu tới 0.7·w/2, cộng với nở thì phải hạ thêm): chủ yếu `valve_gate` ở w = 2.
- **pixelate 1.0 → 0.5** (riêng lẻ không hỏng; hạ vì cộng với nở + blur: `--tune` hạ cả ba cùng 0.4× phần còn lại, rồi làm tròn xuống 0.4 / 0.35 / 0.5).
- xoay, nghiêng, ánh sáng, nhiễu, JPEG: **không phải hạ** (mỗi cái riêng và cả đoạn tổng đều đạt).

**Kết quả sau khi hạ** (k ≥ 3.5, 288 phép/kịch bản): **0 thất bại ở mọi kịch bản**, kể cả mọi tham số cùng mạnh nhất. k = 3.0 (thông tin): 1 thất bại ở kịch bản mòn riêng (`valve_needle` X8076 → `valve_general` 2101, w = 4, margin −0.002). 2 trường hợp "trượt chặt" (khớp đúng lớp nhưng sang tệp khác cùng lớp: `valve_butterfly` hai bản vẽ X8075_1 / X8075_2) không coi là lỗi.

**Mẫu bị loại khỏi phép thử** (đáng nêu): ở bản **sạch**, một số cặp lớp gần như đồng nhất theo NCC (margin < 0.03 = ngưỡng ồn: nhiễu hoặc JPEG riêng chỉ làm margin đổi tối đa 0.026): 50/384 tổ hợp (template × k × w): `check_valve_swing` X8078 (12), `check_valve_globe` X2113 (12) ↔ `valve_globe` X8068 (7), `valve_butterfly` X8075_1 (9), `valve_gate` X8074 (4 — ở k ≤ 3.5, w ≥ 3), `valve_three_way` 2103 (3), `valve_three_way_globe` X8070 (3). Với các tổ hợp này không kết luận gì từ suy giảm (khác biệt giữa chúng nhỏ hơn ngưỡng ồn ngay cả khi sạch); chúng **vẫn** nằm trong CSV (`testable = 0`). Nghĩa là `valve_globe` vs `check_valve_globe` (chỉ khác một vạch nhỏ) có thể dễ nhầm với nhau trong thực tế ở mọi cỡ k — đó là giới hạn của NCC / của bản vẽ, không phải của V7.

**Bổ sung sau review (2026-11-08): kiểm vùng khác biệt cho 50 tổ hợp trên.** Review V7 mục 2.1 chỉ ra các cặp
này thật ra là `valve_globe` ↔ `valve_butterfly` (X8075_1) và `check_valve_globe` ↔ `check_valve_swing`
(câu trên ghi nhầm cặp); hai lớp chỉ khác **một vạch đứng xuyên chấm giữa** (~5% mực), nên NCC cả ký hiệu bị
phần chung lấn át (làm mờ vạch từ 100% về 0% chỉ đổi NCC ~0.02). Script nay kiểm riêng **vùng khác biệt**
lấy từ hai template (`detail_core`: mực của lớp này ở chỗ lớp kia là giấy, lấy đường giữa mỗi mảnh ≥ 4 px):
- lõi chi tiết của chính nó phải đậm hơn mức giữa hai bản sạch (gần bản sạch của mình hơn);
- chỗ chi tiết của lớp kia phải nhạt hơn mức giữa (không "mọc" chi tiết của lớp kia).

Cột mới trong `survival.csv`: `detail_vs`, `detail_own`, `detail_other`, `detail_pass` (trống khi template
kiểm được bằng NCC). Các cột cũ trùng y lần chạy trước. Kết quả (`survival.txt`): **600/600 đạt** (50 tổ hợp ×
12 kịch bản); trong cổng k ≥ 3.5, lõi chi tiết còn ít nhất **0.53** (`valve_butterfly`, mòn tối đa, k = 3.5,
w = 3), chỗ chi tiết lớp kia đậm nhiều nhất **0.43** (`valve_three_way` so với `_globe`, nở tối đa, k = 3.5,
w = 4: giữa ký hiệu sẫm lại như có chấm; ca sát ngưỡng nhất). Dưới cổng (k = 3.0) chấm của
`valve_three_way_globe` nằm trong 1 px nét của `valve_three_way`, nên ngay bản sạch đã đo 0.71 ở đó: ngưỡng
tính từ bản sạch nên vẫn đúng, nhưng phép đo kém nhạy ở cỡ này. Test: `test_the_detail_core_is_the_middle_of_
what_only_one_drawing_has`, `test_whole_symbol_ncc_barely_sees_the_bar_the_detail_check_does`.

Bản đồ nhiệt margin (`ncc_summary.png`): xanh = còn cách lớp khác xa; cam/đỏ nhạt = margin gần 0 từ khi sạch (cùng các cặp trên).

## 4. Hình học đi theo ảnh

**Affine** `M` (2×3, sạch → suy giảm, toạ độ trang kiểu SVG/COCO: gốc ở góc trên-trái của pixel trên-trái, tâm pixel ở +0.5; xoay quanh tâm trang, nghiêng ngang trước xoay). Ghi **chính xác** (float, không làm tròn) trong `affine` của JSON: `x' = a·x + b·y + tx`. Cùng một `M` đưa vào `cv2.warpAffine` (sau khi đổi ±0.5) và dời mọi toạ độ. Test: đặt chấm tại điểm biết trước, xoay/nghiêng 4 cặp góc, tâm chấm trên ảnh khớp `apply_affine` (≤ 0.35 px).

Trong JSON mỗi ảnh suy giảm: `annotations` (`bbox` = hộp mực đã dời; `bbox_clean`; `bbox_corners` = cách dời 4 góc, để so), `nodes` (mọi nút của sheet V4: `box` = hộp 4 góc dời, `center`, `ink_boxes`; **nút bong bóng lấy hộp của vòng tròn đã dời** = bbox annotation, không phải hộp PID2Graph), `ports`, `stems` (đã dời).

**Hai cách dời hộp** — đo (`check_degrade_geometry.py`, 7 tờ × 2 biến thể × (+1°, +0.5°) và dấu ngược = 3706 annotation, đối chiếu với hộp đo trên **ảnh đã xoay thật** bằng `cv2.warpAffine` mặt nạ ink của riêng ký hiệu; `geometry.txt`, `geometry_errors.csv`):

| loại | n | IoU trung vị: 4 góc / mực | sai lệch cạnh lớn nhất px: 4 góc / mực |
|---|---|---|---|
| ký hiệu ISO | 1152 | 0.948 / 0.969 | 2.5 / 0.9 |
| mũi tên | 904 | 0.925 / 0.948 | 2.4 / 0.9 |
| bong bóng | 856 | 0.988 / 0.988 | 1.5 / 0.9 |
| `flanged_connection` | 274 | 0.892 / 0.943 | 1.8 / 0.9 |
| đường dẫn (leader) | 172 | 0.862 / 0.949 | **11.5** / 0.9 |
| bình | 80 | 0.993 / 0.997 | **8.3** / 0.9 |
| bình cao (cao/rộng ≥ 2) | 36 | 0.992 / 0.996 | **6.2** / 0.9 |
| `column` | 12 | 0.996 / 0.997 | 3.0 / 0.9 |
| agitator | 8 | 0.993 / 0.997 | 1.8 / 0.9 |

**Chọn: hộp theo mực** (`ink_box_after`): cạnh lệch tối đa 0.9 px (do nội suy nhị phân khi xoay ảnh), cách 4 góc lệch tới 11.5 px (leader dài chéo) / 8.3 px (bình) / 6.2 px (bình cao). Với xoay 1° một vật cao 3000 px thì 4 góc phình thêm ~50 px nếu vật không lấp đầy hộp (bình có đầu bầu). Lưu ý: ở vật lấp đầy hộp (thanh trục agitator) hai cách như nhau — test dùng elip để thấy khác biệt.
Chống rò: phần mực ký hiệu **ngoài mọi hộp đã dời** = 1.42% (trung bình, max 3.1%) trên 28 trang xoay, so với 1.26% (max 2.8%) khi không xoay (khác biệt do mực không có annotation: đường tín hiệu, đuôi leader) → xoay không làm hộp hụt thêm đáng kể. "Mực trong hộp": ≥ 0.86–0.99 (nội suy 1 px trên vật nhỏ).
Hộp ra ngoài trang: 0/413 ảnh (xoay 1° trên trang ~7000 px, lề trắng đủ).

Ảnh `bbox_overlay.png` (đã xem): lò phản ứng tờ 9 `v012`, xoay 0.98°, nghiêng 0.35°: khung đỏ (mực) khít đầu bầu và trục/cánh khuấy của agitator, khung xanh (4 góc) chồng lên hầu như trùng ở bình này (bình lấp đầy hộp) — khác biệt rõ chỉ ở leader và ký hiệu có hộp chéo.

## 5. Chạy cả bộ (`make_degraded.txt`, `degraded.csv`)

`.venv/bin/python scripts/make_degraded.py --workers 8` → 413 ảnh, 93 s (8 tiến trình, 10 lõi). Mỗi ảnh trung bình 1.61 s: vẽ lại trang 0.46, suy giảm 1.09, dời hình học 0.05.
- **Đĩa**: `data/iso_open100/degraded/` = **1.7 GB** (JPEG trung bình **4.2 MB**, 0.4–17.2 MB; JSON 131 KB). Gấp ~13 lần bộ sạch (127 MB). Nhiễu σ lớn làm JPEG phình; ảnh ~7000×4600.
- Tham số thực: xoay −1…1°, nghiêng ±0.5°, morph −1.82…+0.73 px (trung bình −0.45), blur 0…0.68 px, block 1…1.48, nhiễu σ 0.07…9.9, JPEG q 55–90.

**Bố cục để V8 chọn**: `data/iso_open100/variants/sheet<i>/v<NNN>.{png,svg,json}` = sạch (V6, không đổi); `data/iso_open100/degraded/sheet<i>/v<NNN>.{jpg,json}` = suy giảm, **cùng tên** với biến thể sạch tương ứng. V8 chọn sạch, suy giảm hoặc cả hai theo tên; dùng `annotations`/`nodes`/`affine` của file `degraded/…json` cho ảnh suy giảm, của `variants/…json` cho ảnh sạch.

## 6. Ảnh (cục bộ; `*.png` trong `docs/experiments/` bị gitignore)

- `docs/experiments/iso_render/v7/before_after.png` — 2 ký hiệu: sạch / mức nhẹ nhất (chỉ JPEG 90) / mạnh nhất mòn / mạnh nhất nở (đã xem: hai ảnh mạnh thấy rõ nhiễu giấy, mờ, nét mỏng hoặc dày; trục van, dấu giao vẫn còn).
- `docs/experiments/iso_render/v7/thin_details.png` — agitator (trục), cặp mặt bích 511, van `valve_gate` ở mức mạnh nhất, mòn và nở (đã xem: hai vạch mặt bích vẫn tách rời ở cả hai chiều; trục agitator mòn còn là đường xám mảnh; vạch ngang của van chắn còn).
- `docs/experiments/iso_render/v7/bbox_overlay.png` — hộp mực (đỏ) và hộp 4 góc (xanh) trên trang xoay (đã xem, mục 4).
- `docs/experiments/iso_render/v7/ncc_summary.png` — margin NCC theo template × kịch bản (đã xem, mục 3).
- CSV/TXT: `survival.csv` (+`survival.txt`, `tune.txt`, `survival_before_tuning.txt`), `geometry_errors.csv`, `geometry.txt`, `thin_line.txt`, `degraded.csv`, `make_degraded.txt`.

## 7. Hạn chế

- Trần dilation/blur/pixelate được đặt theo template yếu nhất ở k nhỏ; các tờ ảnh thật dùng k lớn hơn nên **có thể hơi thận trọng**. Nhiễu/JPEG/xoay/ánh sáng ở mức tôi chọn, không template nào chặn.
- Không mô phỏng: nếp gấp, bóng cạnh sách, mực chảy, đốm, kích thước ảnh khác — chỉ là "quét sạch có hơi nhiễu".
- Dải mòn khá yếu ở w = 2 (peak 0.58): đủ cho NCC nhưng một mạng nhận dạng thực có thể thấy khác.
- Chưa đo tác động lên detector Stage 1 (việc của V8 trở đi); chưa có phép thử "mạng vẫn nhận ra" ngoài NCC
  và phép đo vùng khác biệt (mục 3).
- JPEG 4 MB/ảnh: nếu đĩa không đủ có thể chỉ lưu tham số (JSON) và sinh lại ảnh khi cần — kết quả xác định theo seed (test).
- 3706 annotation đo hình học ở 7/12 tờ, 2 biến thể mỗi tờ; không đo mọi biến thể.

## 8. Câu hỏi cho user

1. "Kernel erosion ≤ w/2" tôi hiểu là **bề rộng nét đổi ≤ w/2** (ràng buộc thêm: nét còn ≥ 55% độ đậm). Hiểu theo nghĩa đen (kernel ≤ w/2 px, tức đổi bề rộng ≤ w/2 − 1) sẽ gần như không mòn được ở w = 2 — đúng ý chưa?
2. Chấp nhận trần dilation 0.4·w/2 / blur 0.35·w/2 / pixelate 0.5 (hạ từ plan vì `valve_gate`, `valve_general`, `valve_needle`, `check_valve_general`, `valve_three_way` ở k nhỏ)? Hay muốn giữ trần plan và loại hẳn k < 4–5 khỏi phép thử?
3. Hộp theo mực (khít, sai lệch ≤ 0.9 px) hay 4 góc (đơn giản, phình tới 11 px)? Hiện lưu cả hai trong JSON, `bbox` là hộp mực.
4. Một ảnh suy giảm/biến thể (1.7 GB) có ổn, hay chỉ lưu tham số và sinh lại khi V8 cần?
