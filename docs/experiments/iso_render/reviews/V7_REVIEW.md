# Review V7: làm suy giảm toàn ảnh

- Commit được review: `4a2ed29 feat: RENDER_ISO V7 degradation` (agent Sonnet thực hiện, nằm trên V6 `0c8ae21`).
- Người review: Claude (Opus), ngày 2026-11-06.
- Phạm vi:
  - đọc toàn bộ `src/utils/iso_degrade.py`, phần sửa trong `iso_variants.py`, config, report và plan;
  - chạy test trên worktree sạch tại commit này;
  - tự kiểm lại hình học trên một ảnh đã suy giảm;
  - xem ảnh bằng chứng.

## Kết luận

**Đạt.** Đây là bước làm tốt nhất trong ba bước:

- Affine được lưu chính xác và dùng chung một ma trận cho cả ảnh lẫn toạ độ.
- Box theo mực khít với ảnh đã xoay.
- Bong bóng lấy tâm từ vòng tròn đã dời.
- Giới hạn suy giảm theo w có thêm một ràng buộc đo được (`MIN_PEAK`).
- Trần tham số được hạ dựa trên phép đo.

Các điểm còn lại chủ yếu là giới hạn của phép kiểm NCC và chi phí đĩa, không phải lỗi.

## 1. Những gì đã kiểm và đạt

| Kiểm | Cách kiểm | Kết quả |
|---|---|---|
| Test | worktree tách tại `4a2ed29`, có OPEN100 và template | **548 passed** (533 + 15 mới) |
| Ruff | `ruff check` trên 6 file mới/sửa | sạch |
| Không đổi V3–V6 | `STAGES` thêm `degrade`; seed các khâu khác dẫn xuất theo tên khâu nên không đổi. Agent vẽ lại 413/413 trang từ seed và so với PNG sạch | trùng từng pixel (theo report; test riêng so chữ ký và seed với JSON V6) |
| Affine | đọc `affine_for`, `_cv_matrix` (bù ±0.5 px giữa quy ước pixel và quy ước toạ độ trang), `apply_affine` | đúng; cùng một ma trận cho `warpAffine` và cho mọi điểm |
| Tâm bong bóng trong `nodes` | tự kiểm trên `sheet3/v004` (xoay −1.00°, nghiêng −0.33°, ảnh xoay mạnh nhất cả bộ) | **26/26** bong bóng: tâm nút trùng tâm bbox COCO đã dời (lệch < 1 px) |
| Box mực khít ảnh đã suy giảm | cùng ảnh, 4 annotation cao nhất (2 bình, 1 `general` lớn, 1 bơm): mực của chính ảnh JPEG chạm cả 4 cạnh box | đạt; box 4 góc rộng hơn tối đa 2 px ở ca này |
| Ảnh bằng chứng | đã xem `thin_details.png`, `bbox_overlay.png` | đúng như report mô tả: hai vạch mặt bích vẫn tách rời khi mòn và khi nở; trục agitator khi mòn mạnh nhất còn là nét xám mảnh nhưng liền; box đỏ khít đầu bầu và trục khuấy |
| Ràng buộc | xem diff | không đụng Stage 1/2, `config.yaml`, `pyproject.toml`, `third_party`; `degrade()` viết lại, không import |

## 2. Vấn đề và góp ý

### 2.1 [Trung bình] Phép kiểm "chi tiết còn sống" bằng NCC không phân biệt được đúng những cặp dễ nhầm nhất

- **50/384 tổ hợp** bị loại khỏi phép thử (`testable = 0`), vì ngay ở ảnh **sạch**, NCC giữa hai lớp đã gần như bằng nhau (chênh lệch < 0.03).
- Các template bị loại là hai cặp (sửa 2026-11-08: bản đầu của review ghi sai cặp):
  - `valve_globe` (X8068) ↔ `valve_butterfly` dạng 1 (X8075_1): NCC sạch 0.973;
  - `check_valve_globe` (X2113) ↔ `check_valve_swing` (X8078): NCC sạch 0.975.
- Hai lớp trong mỗi cặp chỉ khác nhau **một vạch đứng xuyên qua chấm giữa**, chiếm khoảng 5% mực (`img/v7_21_X8075_1_X8068.png`, `img/v7_21_X8078_X2113.png`). Kết luận "0 thất bại" vì thế **không bao phủ** chúng.
- Nguyên nhân: NCC chấm điểm cả ký hiệu, nên phần chung (hai tam giác, chấm) chiếm gần hết điểm. Làm mờ dần vạch từ 100% về 0% chỉ làm NCC đổi 0.022–0.024 (`img/v7_21_fade.png`). Sau suy giảm tối đa, NCC đúng/nhầm chỉ còn cách nhau khoảng 0.01, trong khi trên ảnh vạch vẫn thấy rõ. Nghĩa là NCC không đủ nhạy để biết vạch còn hay mất.
- **Đề xuất kiểm bổ sung:** đo riêng vùng khác biệt giữa hai template.
  - Vùng vạch = mực của template có vạch, ở chỗ template không vạch là giấy trắng.
  - Sau suy giảm tối đa, lõi vạch (độ đậm đỉnh theo từng hàng, cách `MIN_PEAK` đo nét) phải còn đậm.
  - Trên template không vạch, cùng chỗ đó phải vẫn nhạt (không sinh vạch giả).
- **Đo thử** (3 cỡ k × 3 nét w × 2 cặp × 11 kịch bản = 198 tổ hợp, `v7_21_bar_trial.csv`):
  - lõi vạch còn đậm ít nhất **0.51** (sạch 1.00); chỗ vạch trên template không vạch đậm nhiều nhất **0.25** (nền có nhiễu và ánh sáng lệch); chênh lệch nhỏ nhất **0.34**. Vạch luôn còn phân biệt được.
  - 4/198 tổ hợp có lõi vạch 0.51–0.54, hơi dưới `MIN_PEAK` 0.55, đều ở `all_erode` (mọi tham số cùng ở đầu mạnh nhất, kể cả mòn). Khi sinh thật, các tham số được bốc ngẫu nhiên đều trong khoảng, nên gần như không bao giờ cùng chạm đầu mạnh nhất.
  - Kết luận tạm: đây là **lỗ hổng của bằng chứng**, không phải lỗi dữ liệu đã thấy. Nên đưa phép đo này vào `check_degrade_survival.py` để bằng chứng của V7 bao phủ cả hai cặp.
- **Trạng thái (2026-11-08): đã sửa.** `check_degrade_survival.py` kiểm vùng khác biệt cho mọi tổ hợp
  `testable = 0`. Cặp được chọn tự động, là các lớp có NCC sạch trong 0.03; vùng lấy từ hai template; luật đạt
  là "gần bản sạch của mình hơn", như phép NCC nhưng chỉ trên vùng khác biệt. Kết quả: 600/600 đạt; trong cổng
  lõi chi tiết ≥ 0.53, chi tiết lớp kia ≤ 0.43 (ca sát nhất: `valve_three_way` nở tối đa). Các cột NCC cũ
  không đổi. Thêm 2 test. Chi tiết: V7 REPORT mục 3.
- **Các công trình khác làm thế nào** (tra 2026-11-08):
  - Cuộc thi nhận dạng ký hiệu GREC (2003, 2005, 2007, 2011) làm suy giảm ký hiệu bằng mô hình Kanungo theo từng **mức** và báo cáo tỉ lệ nhận đúng theo từng mức. Họ không bảo đảm chi tiết còn sống, mà đo xem hiệu năng giảm tới đâu (giảm mạnh ở mức suy giảm cao).
  - Kanungo và cộng sự (TPAMI 2000) kiểm **độ thật** của mô hình suy giảm bằng kiểm định thống kê (hoán vị) so với ảnh quét thật, không kiểm chi tiết phân biệt lớp. Họ lưu ý: so ma trận nhầm lẫn của OCR thì chỉ kiểm được cặp "mô hình + OCR", không kiểm được mô hình suy giảm.
  - Các bộ sinh ảnh tài liệu như DocCreator (2017) và Augraphy (ICDAR 2023) chủ yếu nhắm ảnh văn bản; trong mô tả của họ không thấy bước kiểm chi tiết phân biệt lớp.
  - Bên phân loại mịn (fine-grained), Pu và cộng sự (2023) và SnapMix (AAAI 2021) nêu đúng vấn đề này: phép tăng cường có thể xoá hoặc tráo vùng phân biệt, làm nhãn sai. SnapMix ước lượng vùng phân biệt bằng CAM; Pu và cộng sự chuyển việc tăng cường sang không gian đặc trưng. Ta có lợi thế là biết **chính xác** vùng phân biệt từ hai template, không cần ước lượng.
  - Các bộ dữ liệu P&ID tổng hợp (Dataset-P&ID của Digitize-PID, SynthPID) không báo cáo phép kiểm nào như vậy; SynthPID không có bước suy giảm.
  - Gợi ý cho luận văn: ngoài phép kiểm trên, báo cáo độ chính xác Stage 2 theo **mức suy giảm** (chia ảnh theo tham số đã lưu trong JSON) như GREC.

### 2.2 [Thấp] Ngưỡng gate k ≥ 3.5 px/mm bỏ ngoài khoảng 5% van

- Van G1 có p5 = 3.2 px/mm, nên khoảng 5% van thật nằm dưới ngưỡng và không được kiểm.
- Ở k = 3.0 có đúng 1 ca trượt (`valve_needle` → `valve_general`, chênh lệch −0.002, chỉ khi mòn riêng). Không đáng hạ trần thêm, nhưng nên ghi vào luận văn như một giới hạn đã biết.
- **Trạng thái (2026-11-08): bỏ qua, không ảnh hưởng.** Đo lại trên bộ hiện tại (8,604 van G1/G2, k suy từ box mực
  và cỡ mực template): 9.3% có k < 3.5 và 5.4% có k < 3.0 (nhiều hơn "khoảng 5%" ghi trên). Nhưng **toàn bộ** số
  này thuộc ba lớp `valve_control` (64% mẫu của lớp), `safety_valve_spring` (46%), `safety_valve_spring_angle`
  (65%): template cao (bộ truyền động, lò xo) nên vẽ ít px/mm hơn. Đây là những lớp khác biệt rõ nhất (NCC sạch
  chênh 0.15–0.84). Chạy phép kiểm ở k = 2.3 / 2.5 / 3.0 (p1 của bộ = 2.29): ba lớp này **0 lần trượt**, chênh
  nhỏ nhất 0.074. Các ca trượt ở k ≤ 2.5 chỉ thuộc lớp không bao giờ bị vẽ nhỏ thế (`valve_needle`,
  `valve_butterfly`, ... đều 0% dưới 3.5). Với luận văn: ghi "cổng k ≥ 3.5; phần dưới cổng là các lớp dễ phân
  biệt, đã kiểm tới k = 2.3". Nếu muốn báo cáo kỹ hơn, chia mAP theo cỡ như COCO (nhỏ / vừa / lớn).

### 2.3 [Thấp] Chi phí đĩa và mức nhiễu

- 1.7 GB JPEG (trung bình 4.2 MB, tối đa 17.2 MB), gấp 13 lần bộ sạch. Phần lớn do nhiễu Gauss σ tới 10 làm JPEG nén kém.
- Ảnh quét thật thường ít nhiễu Gauss mà nhiều đốm hoặc vệt hơn.
- Hai lựa chọn:
  - hạ trần `noise_sigma` (ví dụ 6) để JPEG nhỏ lại, và thêm đốm hoặc vệt nếu muốn giống ảnh quét hơn;
  - chỉ lưu tham số và sinh lại ảnh khi cần (kết quả xác định theo seed, đã có test).
- Nếu ổ đĩa còn chỗ thì cứ giữ nguyên.
- **Trạng thái (2026-11-08): bỏ qua, không ảnh hưởng tới kết quả.**
  - Dung lượng (nay 432 ảnh, 4.1 MB/ảnh) chỉ là chi phí lưu và chép lên Colab, không đổi số đo.
  - Kiểu nhiễu: GREC dùng mô hình Kanungo (đảo pixel gần mép nét, ảnh nhị phân); Augraphy / DocCreator mô phỏng vết
    in, quét, mực loang. Nhiễu Gauss của ta kém thật hơn, nhưng với detector chạy theo tile và crop Stage 2 nét
    3 px thì σ ≤ 10 mức xám (≈ 4% độ đậm) là nhỏ so với các chênh lệch đã đo ở 2.1 (≥ 0.34). Thay mô hình nhiễu
    không đáng công so với khoảng cách lớn hơn ở 2.5 (chữ).

### 2.4 [Thấp] Box của nút GraphML dùng cách dời 4 góc

- `nodes[*].box` là box 4 góc đã dời, rộng hơn thực tế tới khoảng 8 px với bình. `center` thì chính xác, và `ink_boxes` (box mực) cũng có sẵn.
- Nếu V8 hoặc bước đánh giá đồ thị so khớp nút bằng IoU của box thì nên dùng hợp của `ink_boxes`. Nếu so bằng tâm thì không ảnh hưởng.
- **Đo lại (2026-11-08) trên 432 ảnh V7: ý trên đúng nhưng chỉ là nguyên nhân phụ.** `nodes[*].box` là box nút
  GraphML gốc của PID2Graph (đã nhân `size_factor` rồi xoay), không phải box ký hiệu đã vẽ. Ảnh:
  `img/v7_24_corners.png`, `img/v7_24_real.png`.
  - Phụ, do dời 4 góc: so `bbox_corners` với `bbox` của cùng annotation, box nở ≤ 1.5 px ở 95% ký hiệu ISO, tối
    đa 8.6 px (leader tối đa 14 px). IoU ≥ 0.81 với ký hiệu ISO.
  - Chính, do ký hiệu ISO khác hình / khác chỗ so với box gốc: IoU giữa `box` và hợp `ink_boxes` của nút ISO có
    trung vị 0.49 (50% dưới 0.5). Ví dụ van điều khiển: box gốc gồm cả bộ truyền động, ký hiệu ISO thì không. V5
    còn dời một số ký hiệu ra khỏi chỗ cũ (`general61` ở tờ 2: IoU 0). Bình, cột lớn thì khớp (IoU 0.99).
  - Vì vậy `center` chỉ đúng là tâm box gốc: với nút ISO, nó nằm ngoài box mực ở 28% trường hợp. Bong bóng, cờ,
    generic thì khớp (tâm lấy từ vòng tròn đã vẽ).
- **Trạng thái (2026-11-08): bỏ qua, không ảnh hưởng.** Stage 1 / Stage 2 chỉ đọc COCO (box mực, đúng). V8
  (`iso_export.node_geometry`) ghi box nút vào GraphML xuất ra bằng hợp box mực của các annotation; `nodes[*].box`
  chỉ là dự phòng cho nút không vẽ gì (connector, crossing, background). `evaluation.py` chỉ tính mAP trên COCO.
  Nếu sau này đánh giá đồ thị kiểu PID2Graph (ghép nút bằng IoU) thì đọc box từ GraphML V8 hoặc hợp `ink_boxes`,
  **không** dùng `nodes[*].box` hay `center` của JSON V7.

### 2.5 [Ghi nhận] Độ thật của ảnh suy giảm

- Mô hình suy giảm là "bản quét sạch có nhiễu": ánh sáng lệch tối đa 0.2, không có nếp gấp, đốm, mực loang hay chữ.
- Khoảng morph [−1, 0.4] lệch về phía mòn (trung bình −0.45 px), nên bộ ảnh nghiêng về nét mảnh. Đây là hệ quả của việc hạ trần nở để van không bị nhầm, đã được giải thích trong report.
- Report đã nêu tất cả các điểm này trong mục hạn chế.
- **Trạng thái (2026-11-08): nếp gấp / đốm / mực loang: bỏ qua (như 2.3). Thiếu chữ: có ảnh hưởng, cần user
  quyết; không thuộc phạm vi V7.**
  - Ảnh của ta **không có chữ**: bong bóng ISA là vòng tròn rỗng, không có số hiệu ống hay tên thiết bị (plan V2:
    "chữ để v2"). Bản vẽ thật, Dataset-P&ID (DigitizePID, nơi Stage 1 và encoder Stage 2 được train) và SynthPID
    (`renderers.py` vẽ chữ) đều có chữ.
  - Stage 1: chữ là nguồn dương giả lớn ở bản vẽ thật; test không có chữ thì precision bị thổi phồng, và nếu train
    trên bộ này thì detector chưa từng thấy chữ.
  - Stage 2: crop `other_isa` là vòng tròn rỗng, trong khi bong bóng thật (và trong DigitizePID) có chữ bên trong.
    Cổng "Other" vì thế được đo trên một kiểu mẫu dễ hơn / khác với thật.
  - Hướng xử lý (nếu làm): thêm bước vẽ chữ ở V5 (mã tag trong bong bóng, nhãn ống cạnh đường ống, sinh ngẫu
    nhiên; chỉ mượn ý từ `annotate.py` của SynthPID, tự viết), không gán nhãn COCO cho chữ, phải giữ G1 (chữ
    không đè ký hiệu). Tối thiểu: ghi rõ giới hạn này khi báo cáo số Stage 1 và "Other".
  - **Trạng thái (2026-11-08): đã làm (V5c).** Chữ là lớp riêng, chỉ đặt vào chỗ trống, không vào COCO: mã ISA
    trong bong bóng, số hiệu ống, tag thiết bị, mã trong cờ, nhãn ở đầu đường dẫn. G1 thêm tiêu chí 8 (chữ không
    chạm mực). Tiêu chí 1–7 y nguyên 24/24 lượt; 432/432 SVG = SVG cũ + nhóm chữ. Chi tiết: plan mục V5c.

## 3. Câu hỏi agent nêu, kèm ý kiến của tôi

| # | Câu hỏi | Ý kiến review |
|---|---|---|
| 1 | "Kernel erosion ≤ w/2" hiểu là bề rộng nét đổi tối đa w/2, kèm ràng buộc `MIN_PEAK` 0.55 | **Đồng ý.** Hiểu theo nghĩa đen thì gần như không mòn được ở w = 2; ràng buộc theo độ đậm còn lại là cách đo đúng bản chất |
| 2 | Hạ trần nở 0.4, blur 0.35, pixelate 0.5 | **Chấp nhận.** Các trần này có số đo trước/sau trong `tune.txt` |
| 3 | Box theo mực hay theo 4 góc | **Theo mực** (lệch ≤ 0.9 px, so với tối đa 11.5 px nếu theo 4 góc). Giữ cả hai trong JSON là hợp lý |
| 4 | 1.7 GB | Xem 2.3: tuỳ dung lượng đĩa; có thể hạ σ nhiễu hoặc sinh lại khi cần |

## 4. Việc nên làm (không chặn V8)

1. ~~Đưa phép đo vùng khác biệt (vạch) của các cặp dễ nhầm vào `check_degrade_survival.py` (2.1).~~ Đã làm 2026-11-08: 600/600 đạt.
2. ~~Quyết về σ nhiễu và cách lưu ảnh (2.3).~~ Bỏ qua (2026-11-08): không ảnh hưởng số đo.
3. ~~V8 dùng hợp `ink_boxes` nếu cần IoU nút (2.4).~~ Bỏ qua: V8 đã dùng hợp box mực; `nodes[*].box` chỉ dự phòng cho nút không vẽ.
4. ~~Quyết có thêm chữ vào ảnh hay không (2.5).~~ Đã làm 2026-11-08 (plan V5c).
