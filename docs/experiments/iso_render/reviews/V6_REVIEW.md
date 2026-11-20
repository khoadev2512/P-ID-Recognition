# Review V6: sinh nhiều biến thể và lọc trùng

- Commit được review: `0c8ae21 feat: RENDER_ISO V6 variants and dedup` (agent Sonnet thực hiện).
- Người review: Claude (Opus), ngày 2026-11-06.
- Phạm vi: đọc toàn bộ diff (code, config, `check_g1.py`, test, plan), chạy lại test trên worktree sạch tại commit này, kiểm tái lập, đo lại các số liệu chính, xem ảnh bằng chứng.

## Kết luận

**Đạt về kỹ thuật, nhưng có một vấn đề thiết kế cần user quyết trước V8.**

- Code gọn, đúng phạm vi: không sửa V3/V4/V5, không đụng Stage 1/2 hay `pyproject.toml`.
- Seed tách theo từng khâu và tái lập đúng từng byte.
- Report trung thực, tự nêu hạn chế.
- Vấn đề chính: **bộ lọc trùng ảnh đang loại các biến thể thật sự khác nhau về ký hiệu** (mục 2.1).

## 1. Những gì đã kiểm và đạt

| Kiểm | Cách kiểm | Kết quả |
|---|---|---|
| Test | worktree tách tại `0c8ae21`, có OPEN100 và template | **533 passed** (516 cũ + 17 mới), 0 skip |
| Ruff | `ruff check` trên 5 file mới/sửa | sạch |
| Tái lập | sinh lại bằng `plan_variant` + `render_variant` 4 biến thể (tờ 4: v000, v058; tờ 9: v000, v035), so với file đã lưu | SVG **trùng từng byte**, annotation trùng |
| `check_g1` sau khi tách `evaluate()` và đổi cách đo tiêu chí 2 | so kết luận PASS/FAIL với bản trước V6 trên 24 lượt (12 tờ × seed 0–1) | **trùng hoàn toàn**, kể cả 2 lượt trượt cũ ở tờ 10 |
| `evaluate()` nhận `runs4` thay cho `runs` | tiêu chí 5 chỉ dùng cặp nút (`a`, `b`, `internal`) của các run, không dùng toạ độ | không đổi kết quả (khớp phép so trên) |
| Ràng buộc | xem diff | không đụng `config.yaml`, Stage 1/2, `third_party`, `pyproject.toml` |
| Ảnh bằng chứng | đã xem `montage_sheet3.png`, `style_crops.png`, `duplicate_pair_0_sheet0.png` | đúng như report mô tả. Ở w = 2/3/4, nét ống và nét ký hiệu dày lên cùng nhau; ở cỡ 0.85/1.15, ống vẫn nối vào ký hiệu |

## 2. Vấn đề

### 2.1 [Quan trọng] Lọc trùng ảnh đang loại mất sự đa dạng ký hiệu

- Hash 16×16 trên ảnh khoảng 7000×4600 px nghĩa là mỗi ô dài khoảng 450 px. Ở độ phân giải đó, hash chỉ thấy **bố cục**, không thấy ký hiệu nào được vẽ.
- Tôi đo lại cả **66 bản bị loại vì trùng**: mỗi bản đều khác bản đã giữ gần nhất ở **15–40% số nút** (trung vị 20%), tính theo lớp, template, xoay, mặt bích hoặc agitator. Không bản nào là trùng thật.
- Ví dụ `duplicate_pair_0_sheet0.png` (khoảng cách 3): hai ảnh cùng bố cục nhưng nhiều van khác lớp, và nhiều bong bóng đã được thay bằng thiết bị G5.
- Hậu quả:
  - Tờ 1 chỉ giữ được 26/36 (31 lần thử bị loại vì "trùng").
  - Với Stage 1/2, phép gán ký hiệu chính là nguồn đa dạng có giá trị nhất, nên bộ lọc đang vứt đúng thứ cần giữ.
- Agent đã nêu điều này trong mục hạn chế, nhưng số thực tế (100% bản bị loại khác nhau ≥ 15% số nút) nặng hơn mức "chấp nhận" mà report đánh giá.
- **Đề xuất**, chờ user chọn:
  - (a) Chỉ coi là trùng khi hash gần **và** phép gán khác dưới khoảng 5% số nút.
  - (b) Bỏ lọc bằng ảnh, giữ lọc chữ ký phép gán. Lọc chữ ký vốn chưa bao giờ kích hoạt, vì 0/552 lần thử trùng chữ ký.

  Tôi nghiêng về (a): vẫn chặn được hai bản gần như giống hệt, mà không mất biến thể khác ký hiệu.

- **Trạng thái (2026-11-06): đã sửa**, theo biến thể (a′) user duyệt: lọc theo **khác biệt từng nút**
  (`variants.min_node_diff: 0.05`), hash ảnh chỉ ghi lại. Không dùng histogram nhãn kiểu SynthPID vì
  histogram không thấy hai ký hiệu đổi chỗ, và biến thể cùng tờ có cùng tập id nút nên so được từng nút.
  Kết quả chạy lại: 428 giữ, 0 trùng, 11/12 tờ đủ 36; khoảng cách nội dung tới bản giữ gần nhất min 19%.
  387 biến thể chung với lần đầu trùng từng byte; V7, V8 đã chạy lại. Chi tiết: mục 0 của V6 REPORT.

### 2.2 [Trung bình] Loại theo tiêu chí 5 làm lệch phân bố cỡ ký hiệu

- Tôi đo tỉ lệ biến thể có cỡ < 0.95, so giữa các lần thử và các bản được giữ:

  | | Lần thử | Bản giữ |
  |---|---|---|
  | Tờ 10 | 45% | 25% |
  | Tờ 4 | 43% | 26% |
  | Tờ 1 | 20% | 8% |
  | Cả bộ | 32% | 28% |

- Report có nói "lệch, chưa đo"; số trên là phần đo bổ sung.
- Nguyên nhân gốc là lỗi V5 đã biết: khi ký hiệu thu nhỏ thì tách khỏi mũi tên hoặc ký hiệu sát bên. V6 chỉ làm nó lộ ra nhiều hơn.
- Có ba hướng:
  - giữ quyết định loại và ghi độ lệch vào luận văn;
  - chuyển sang gắn cờ (`g1: {drop: [1,2,3,6,7], flag: [5]}`);
  - sửa gốc ở V5: mũi tên dời theo ký hiệu mà nó nối. Đây là cách đúng nhất, nhưng nằm ngoài V6.
- **Trạng thái (2026-11-06): đã sửa ở V5** theo hướng C user duyệt: nối khe giữa hai ký hiệu nối thẳng
  bằng một đoạn ống ngắn (`page.bridge_gap: 1.5`), mở rộng luật `_nozzle_stub` sẵn có. Đo thêm cho thấy
  khe có hai nguồn: template hẹp hơn hộp (40/63) và thu nhỏ (23/63). Sau sửa: 432 giữ, 12/12 tờ đủ 36,
  0 lần loại vì tiêu chí 5; tỉ lệ cỡ < 0.95 ở tờ 1, 2, 4, 10 bằng nhau giữa lần thử và bản giữ (34%).
  Giải thích có ảnh: `img/c5_1_mechanism.png` … `c5_5_after.png`; chi tiết: V6 REPORT mục 0b.

### 2.3 [Thấp, tiềm ẩn] Bản `source` đã đổi cỡ được dùng cho cả việc cho phép đè lẫn tiêu chí 3

- `render_variant` đưa bản `source` có hộp ký hiệu đã nhân hệ số vào `render_page`, và `check_g1` cũng dùng bản đó.
- Cả `_resolve_overlaps` của V5 lẫn tiêu chí 3 đều coi "cặp đã đè trong source" là được phép. Về lý thuyết, khi phóng 1.15, các cặp đè **vì bị phóng** có thể lọt qua cả hai lớp.
- Đo trên 24 biến thể có hệ số ≥ 1.10, so với `source` gốc chưa đổi cỡ: **0 cặp bị giấu**. Hiện không gây lỗi.
- Đề xuất: tiêu chí 3 của G1 nên luôn đo với `source` gốc, để lớp kiểm tra không phụ thuộc cách V6 truyền cỡ.
- **Trạng thái (2026-11-08): đã sửa, vì có ảnh hưởng thật.** Đo lại trên cả 235 bản giữ có hệ số > 1 của bộ
  hiện tại (không chỉ 24 bản như lần review đầu): **5 bản** ở tờ 4 có cặp bị giấu (đĩa thay thế cắm xuyên ống
  phía trên; ở v029 hai đĩa đè nhau 64 px). Ảnh hưởng tới huấn luyện / đánh giá: ảnh sai kiểu bản vẽ thật,
  box chồng nhau, và crop Stage 2 của các đĩa này chứa mực ký hiệu khác. Về số lượng thì nhỏ: khoảng 6
  annotation trên ~47k, khoảng 1% mẫu mỗi lớp đĩa (~630 mẫu/lớp). Nhưng đây là lỗi có hệ thống (lặp lại ở cùng một cặp khi phóng to
  đủ lớn), nên sinh thêm biến thể thì sẽ nhiều thêm; cách sửa lại rẻ. Sửa: V5 và G1 cùng đọc cặp
  "vốn chạm" từ sheet gốc (`touch_source`). Đo lại 0/235; thêm lợi ích là 15 bản thu nhỏ không còn bị thu nhỏ /
  thay hộp thừa. Chi tiết: V6 REPORT mục 0c; ảnh `img/r23_*.png`.

### 2.4 [Thấp] Các điểm nhỏ

- Tiêu chí 2 đọc nét w = 3 thành **2.75 px**, tức lệch cố định −0.25 px (phép trừ 1 pixel tâm được làm ở độ phân giải 4×). Tiêu chí này so hiệu giữa hai lớp nên không ảnh hưởng đạt/trượt, nhưng con số trong log dễ gây hiểu lầm. Report đã ghi điều này.
- Annotation của cờ (`kind: flag`) có `label: None`. V8 phải ánh xạ rõ: bỏ, hay gán một nhãn như `other_generic`.
- Bộ lọc `quality` (`bubbles_unsettled`, `unmatched_ports`) chưa kích hoạt lần nào trên dữ liệu thật (0/552). Report đã nói rõ.
- `size_factor` là một hệ số cho cả trang. Report và plan viết "một hệ số cho cả tờ", dễ hiểu nhầm thành cố định theo tờ gốc; thực ra mỗi biến thể có hệ số riêng.
- **Trạng thái (2026-11-08): bỏ qua cả bốn điểm**, vì không điểm nào đổi ảnh hay nhãn mà Stage 1/2 dùng:
  - Tiêu chí 2 đọc 2.75 px: chỉ là con số trong log của bộ kiểm; kết luận đạt/trượt và ảnh không đổi.
  - Cờ có `label: None`: V8 đã bỏ cờ khỏi COCO (quyết định có ghi). Train Stage 1 thì cờ là nền không nhãn, đúng
    vì cờ không có trong vocab; khi đánh giá thì phát hiện trúng cờ tính là dương giả, đã ghi ở review V8 mục 2.6.
    Stage 2 không có crop cờ.
  - Bộ lọc `quality` chưa kích hoạt: không loại ảnh nào, nên không ảnh hưởng dữ liệu.
  - Cách viết "một hệ số cho cả tờ": chỉ là câu chữ trong tài liệu.

## 3. Câu hỏi agent nêu, kèm ý kiến của tôi

| # | Câu hỏi | Ý kiến review |
|---|---|---|
| 1 | `min_distance = 8` | Không chỉ là chuyện ngưỡng, mà là thiết kế: xem 2.1, đề xuất (a) |
| 2 | Tiêu chí 5: loại hay gắn cờ | Tạm giữ **loại**, nhưng ghi độ lệch (2.2) và đưa việc sửa gốc ở V5 vào danh sách việc |
| 3 | 36 biến thể/tờ, `max_attempts` 60 | Hợp lý. Sửa 2.1 thì tờ 1 sẽ đủ 36 |
| 4 | w liên tục hay {2, 3, 4} | Liên tục là tốt; V7 giới hạn suy giảm theo từng w |
| 5 | Đổi cách đo tiêu chí 2 | **Chấp nhận**: kết luận trùng bản cũ ở w = 3, và sửa được trượt giả khi w là số lẻ |

## 4. Việc cần làm trước V8

1. ~~User quyết 2.1. Nếu chọn (a) thì chạy lại `make_variants.py`; khoảng 1 phút.~~ Đã làm 2026-11-06.
2. Ghi độ lệch cỡ (2.2) vào plan và report.
3. (Tuỳ chọn) Tiêu chí 3 đo với `source` gốc (2.3).
