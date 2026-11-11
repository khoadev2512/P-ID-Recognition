# Review V8: xuất dataset và nối vào pipeline

- Commit được review: `c275757 feat: RENDER_ISO V8 dataset export` (agent Sonnet thực hiện, nằm trên V7 `4a2ed29` và V6 `0c8ae21`).
- Người review: Claude (Opus), ngày 2026-11-06.
- Phạm vi:
  - đọc toàn bộ `src/utils/iso_export.py` và `src/pipeline/render_iso.py`, phần sửa trong `pyproject.toml` và config, report;
  - đối chiếu với `build_manifest`, `classes.yaml` hiện có và `vocab.csv`;
  - chạy test trên worktree sạch;
  - tự kiểm topology và tâm bong bóng trên bộ đã xuất;
  - xem ảnh bằng chứng.

## Kết luận

**Đạt về định dạng và tính đúng đắn. Có một khoảng trống so với mục tiêu của plan: lớp thô (coarse) không khớp DigitizePID, nên chưa đánh giá trực tiếp được Stage 1 trên bộ này.** *(2026-11-11: hướng xử lý đã chốt, xem 2.1.)*

Phần ghi file thì chắc chắn:

- GraphML giữ đúng topology.
- Tâm bong bóng trùng tâm box COCO.
- Chia tập theo tờ gốc.
- `data/canonical` không bị đụng; agent có sha256 trước/sau.
- `pyproject.toml` chỉ thêm đúng một dòng.

## 1. Những gì đã kiểm và đạt

| Kiểm | Cách kiểm | Kết quả |
|---|---|---|
| Test | worktree tách tại `c275757`, có OPEN100 và template | **561 passed** (548 + 13 mới) |
| Ruff | 4 file mới | sạch |
| Phạm vi sửa trên cả V6–V8 | `git diff --stat c29a3e6 c275757 -- src/pipeline src/utils pyproject.toml uv.lock configs/config.yaml` | chỉ có file **mới** (`render_iso.py`, `diversity.py`, `iso_degrade.py`, `iso_export.py`, `iso_variants.py`) và **1 dòng** `pyproject.toml`; không sửa module pipeline hay utils có sẵn, `uv.lock`, `config.yaml` |
| Topology GraphML | tự kiểm 40 ảnh chọn ngẫu nhiên (cả sạch lẫn suy giảm): tập nút và tập cạnh so với GraphML gốc OPEN100 | **40/40 khớp**; không có nút agitator |
| Tâm bong bóng (mục checklist V8) | cùng 40 ảnh: tâm nút trong GraphML so với tâm box COCO `other_isa` | **1109/1109** lệch < 0.05 px |
| An toàn ghi | đọc `validate()`: từ chối gốc là `data/` hoặc gốc có `raw/`; chỉ xoá `canonical/images`, `graphml/` và 3 file trong gốc mới | đạt |
| Kiểm chéo V6 | export vẽ lại từng biến thể sạch và so box với JSON đã lưu (sai số 0.02 px), nên nếu V6 được sinh lại bằng config khác thì export sẽ báo lỗi | thiết kế tốt |
| Ảnh bằng chứng | đã xem `overlay_coco.png`, `overlay_graphml.png` | box agitator khít trục và cánh trên cả ảnh sạch lẫn ảnh suy giảm; nút và cạnh GraphML nằm đúng chỗ trên hình |

## 2. Vấn đề và góp ý

### 2.1 [Quan trọng] Lớp thô khác DigitizePID: chưa đánh giá trực tiếp được Stage 1

- `classes.yaml` của gốc mới lấy lớp thô từ `vocab.csv` của Stage 2, gồm 13 lớp: `agitator, blind_disc, check_valve, compressor, fitting, heat_exchanger, motor, other, piping, pump, safety_valve, valve, vessel`.
- DigitizePID, nơi detector Stage 1 được train, có 8 lớp thô: `blind_disc, flange, flow_direction, heat_exchanger, instrument, reducer, safety_valve, valve`.
- Hai tập này lệch nhau: `flow_direction` và `instrument` không còn là lớp thô; `flange` và `reducer` bị gộp vào `fitting`. Vì vậy chạy `detect_infer` rồi `evaluation` trên gốc mới với detector hiện có sẽ cho kết quả sai nghĩa.
- Plan V8 đặt mục tiêu "để `detect_infer` / `fgc_infer` / `evaluation` đọc được". Hiện mới đạt phần **đọc được**, chưa đạt phần **đánh giá được**.
- Agent đã nêu điều này trong mục câu hỏi (số 3), không giấu.
- **Đề xuất:** thêm một tuỳ chọn export, ví dụ `coarse: digitizepid`, ghi lớp thô theo bảng ánh xạ:

  | Lớp ISO | Lớp thô DigitizePID |
  |---|---|
  | `other_isa` | `instrument` |
  | `valve`, `check_valve` | `valve` |
  | `safety_valve` | `safety_valve` |
  | đĩa | `blind_disc` |
  | `flanged_connection` | `flange` |
  | `reducer` | `reducer` |
  | `flow_direction` | `flow_direction` |
  | `heat_exchanger` | `heat_exchanger` |
  | bơm, bình, agitator, `fitting` khác, `other_iso`, `other_generic` | không có tương ứng: bỏ khỏi đánh giá, hoặc gom vào một lớp "ignore" |

- Lớp mịn (cho Stage 2) giữ nguyên như hiện tại. Đây là quyết định của user; nên chốt trước khi chạy đánh giá.

**Đã chốt (2026-11-11): chấm Stage 1 không phân lớp; bảng ánh xạ ở trên chỉ còn là báo cáo phụ, tuỳ chọn. Không làm tuỳ chọn export `coarse: digitizepid`.**

Lý do, từ các quyết định Stage 2 (`docs/PLAN_STAGE2_ONESHOT.md`, "Quyết định 2026-11-09"):

- OPEN100-ISO chỉ làm **đề thi**. Stage 1 không train trên bộ này (ít nhất đến V9), nên detector vẫn là detector DigitizePID 8 lớp thô.
- Trên bản vẽ ISO, nhãn ISO do **Stage 2** gán bằng cách so với 42 prototype. Stage 1 chỉ cần trả lời "chỗ này có ký hiệu".
- Bảng ánh xạ không phủ được phần quan trọng nhất: 26 lớp ISO "H tự nhiên" (bơm, tháp, agitator, …) không có lớp thô DigitizePID tương ứng.

Cách đánh giá Stage 1 trên bộ này:

| Việc | Nội dung |
|---|---|
| Stage 1 không phân lớp | Mọi box GT (kể cả `other_isa`, `other_iso`, `other_generic`) là "ký hiệu"; mọi box dự đoán, dù lớp nào, cũng là "ký hiệu". P, R, mAP@50, mAP@50:95, tách sạch / suy giảm |
| Recall theo nhóm S / V / H tự nhiên / Other | Detector DigitizePID có tìm được hình nó chưa từng thấy không. Đây là trần của cả hệ thống: Stage 1 bỏ sót thì Stage 2 không có gì để nhận |
| End-to-end | TP khi IoU ≥ 0.5 **và** Stage 2 gán đúng nhãn ISO (protocol P4) |
| Bảng ánh xạ 8 lớp thô (ở trên) | Tuỳ chọn, chỉ cho các lớp có tương ứng |

Hệ quả cho Stage 2 one-shot: **không lọc prototype theo lớp thô của Stage 1 trên bản vẽ ISO.** Detector sẽ gán một cái bơm vào lớp thô gần nhất nó biết (vd. `valve`); nếu chỉ so với prototype van thì bơm không bao giờ được nhận đúng. Với P4, so với cả 42 prototype.

Nếu recall nhóm H tự nhiên thấp, đó là một **kết quả** (detector học trên DigitizePID không tổng quát sang hình dạng mới), và là lý do để làm V9: train Stage 1 trên bản vẽ ISO sinh từ 750 sơ đồ tổng hợp, giữ 12 tờ OPEN100 làm test. Augment suy giảm cho Stage 1 (`+detector.aug.degrade.p`) cũng là ứng viên.

Việc code đã đưa vào plan Stage 2: B6a (tắt lọc lớp thô cho P4) và B6b (tuỳ chọn chấm không phân lớp + recall theo nhóm).

### 2.2 [Trung bình] Mặc định `images: both` với `split: all_test` làm tập test trùng lặp

- Mỗi biến thể có mặt hai lần (sạch và suy giảm), lại thêm 36 biến thể cho mỗi trong 12 topology. Chỉ số đo trên 826 ảnh vì thế bị thổi phồng về số mẫu độc lập.
- Đề xuất:
  - khi đánh giá, dùng `images=degraded` (413 ảnh);
  - báo cáo chỉ số **theo tờ gốc**, kèm trung bình trên 12 tờ;
  - đổi mặc định của config thành `degraded`.

  Agent cũng gợi ý điều này.

**Đã chốt (2026-11-11): vấn đề còn, cách sửa đổi.** Protocol P4 cần cả ảnh sạch lẫn ảnh suy giảm để đo độ bền với nhiễu, nên **giữ export `images: both`**, không đổi mặc định. Thay vào đó, quy tắc báo cáo cho mọi số trên bộ này:

- không bao giờ gộp sạch và suy giảm thành một số (tách theo id, ảnh suy giảm có hậu tố `_deg`); **số chính là ảnh suy giảm**, ảnh sạch là tham chiếu;
- báo cáo theo 12 tờ gốc (`iso00`…`iso11`), rồi trung bình ± độ lệch: mẫu độc lập thật chỉ là 12 topology;
- kèm số instance mỗi nhóm.

Ghi trong plan Stage 2, B6b phần P4.

### 2.3 [Thấp] Hai chỗ đặt tên để vừa với code Stage 1 có sẵn

- `file_name` = `synthetic/<stem>`, vì `build_manifest` bắt buộc tiền tố `synthetic/` hoặc `real/`. Nhãn "synthetic" đúng về bản chất (ảnh vẽ lại), và `real/` thì luôn bị xếp vào test.
- File chia tập phải mang tên `digitizepid_splits.json`, vì đó là tên cố định trong `build_manifest`.
- Cả hai chạy đúng và không phải sửa code Stage 1, nhưng tên gây hiểu nhầm. Nên ghi chú trong luận văn hoặc README của data root.

**Đã chốt (2026-11-11):** không sửa code; ghi một câu khi viết phần dữ liệu của luận văn (plan Stage 2, "Ghi chú khi viết phần dữ liệu" cuối B8).

### 2.4 [Thấp] Lớp vocab không có mẫu

- 12/42 lớp vocab có 0 mẫu. Phần lớn là đúng bản chất, vì OPEN100 không có HX, máy nén, motor hay ống mềm.
- `tank` = 0 vì 2062 được xuất thành `other_iso`.
- **`open_disc` = 0:** agent không tìm ra nguyên nhân. Nguyên nhân là taxonomy xếp C0098 `open_disc` vào `scope: extra` (không vẽ được), dù config vẫn liệt kê nó trong `flanges.classes` và trong rule vòi bình. Không phải lỗi export. Nếu muốn có mẫu thì cần đưa C0098 vào `html`, mà dòng taxonomy ghi chú "no stubs: the pipe passes through" nên phải xử lý cổng riêng.

**Đã chốt (2026-11-11): quan trọng hơn trước nhưng hoãn.** `open_disc` (S17) là lớp S và là ứng viên H. Thiếu cổng nối của C0098 cùng gốc với việc thiếu cổng ở cách (d) (prototype có stub), nên gộp vào **Phase C** của plan Stage 2: định nghĩa cổng (user duyệt), đưa vào `html`, sinh lại V6 → V8. Đến lúc đó, nếu S17 nằm trong `h_in` thì P1 vẫn đo được trên DigitizePID, chỉ P4 thiếu lớp này.

### 2.5 [Thấp] Ảnh suy giảm không được kiểm chéo với config V7 hiện tại

- Ảnh sạch được kiểm (vẽ lại rồi so box), nhưng JSON và ảnh `degraded/` được dùng nguyên.
- Nếu V7 được chạy lại với config khác mà V6 không đổi, export sẽ không phát hiện.
- Có thể so `params` trong JSON với `sample_params` sinh lại từ seed.

**Đã chốt (2026-11-11): làm, trước R4.** Ảnh suy giảm giờ là đề thi chính, và V6/V7 đã được sinh lại nhiều lần (2026-11-07/08). Export sẽ sinh lại `sample_params` từ seed với config hiện tại, so với `params` đã lưu, lệch thì báo lỗi và không export. Việc này cũng thay cho quy tắc tay ở §3 câu 4. Plan Stage 2, **B6d** việc 1.

### 2.6 [Ghi nhận] Mũi tên dẫn và cờ không có trong COCO

- Hợp lý, vì cả hai không có trong vocab.
- Hệ quả: detector bắt được chúng thì bị tính là dương giả. Đây là một con số cần ghi khi báo cáo.

**Đã chốt (2026-11-11): quan trọng hơn hẳn, làm trước R4.** Export bỏ **12,096** box loại này (so với 93,800 box được giữ). Khi chấm Stage 1 không phân lớp (2.1), detector DigitizePID có lớp `flow_direction` nên rất dễ bắt mũi tên dẫn, mỗi lần bắt là một dương giả. Cách sửa:

- export ghi các box này thành **vùng bỏ qua** (COCO `iscrowd: 1`);
- `evaluation` không tính dự đoán rơi vào vùng bỏ qua là FP hay TP, GT bỏ qua không vào mẫu số recall; `build_manifest` / `coco_to_yolo` / `extract_crops` bỏ qua chúng;
- báo cáo thêm số dự đoán rơi vào vùng bỏ qua.

Plan Stage 2, **B6d** việc 2.

### 2.7 [Ghi nhận, thuộc V5b] Trục agitator không nhô khỏi đỉnh khi có tag line ngay trên

- Trong ảnh bằng chứng (tờ 9, `v012`, `tank6`), trục bắt đầu đúng ở viền đỉnh, không nhô 3 mm như X8006, vì đoạn nhô sẽ chạm box của đường tag phía trên.
- Đây là hành vi có chủ đích của V5b ("không nhô nếu chạm ký hiệu khác"), không phải lỗi V8. Chỉ nêu để biết trong bộ dữ liệu có cả hai dạng.

**Đã chốt (2026-11-11):** không làm gì. Agitator là lớp "H tự nhiên"; nếu recall agitator thấp bất thường thì xem lại điểm này khi phân tích (plan Stage 2, "Ghi chú khi viết phần dữ liệu").

## 3. Câu hỏi agent nêu, kèm ý kiến của tôi

| # | Câu hỏi | Ý kiến review |
|---|---|---|
| 1 | Cờ và mũi tên dẫn không có trong COCO | Đồng ý; ghi số dương giả khi báo cáo (2.6) |
| 2 | Mặc định `both` + `all_test` | Đổi mặc định sang `degraded` khi đánh giá (2.2). *Cập nhật 2026-11-11: giữ `both`, tách sạch / suy giảm và báo cáo theo tờ gốc khi chấm (2.2)* |
| 3 | Lớp thô khác DigitizePID | **Việc quan trọng nhất còn lại**: cần lớp ánh xạ (2.1). *Cập nhật 2026-11-11: chấm Stage 1 không phân lớp, ánh xạ chỉ là tuỳ chọn (2.1)* |
| 4 | Hardlink giữ nội dung cũ nếu V6 sinh lại mà không export lại | Chấp nhận được, vì export xoá và tạo lại `images/` mỗi lần chạy và kiểm chéo ảnh sạch; ghi vào quy trình: "sinh lại V6/V7 thì phải export lại". *Cập nhật 2026-11-11: export sẽ tự kiểm ảnh suy giảm với config V7 (2.5), quên chạy lại thì báo lỗi* |
| 5 | `by_source` không có tập `val` | Đủ cho mục đích hiện tại (test). Nếu train Stage 1 trên bộ này thì thêm `val` theo tờ gốc. *Cập nhật 2026-11-11: chỉ cần nếu làm V9* |

## 4. Việc nên làm

1. ~~Chốt cách xử lý lớp thô cho Stage 1 (2.1).~~ Đã chốt 2026-11-11: chấm không phân lớp + recall theo nhóm S / V / H; không lọc prototype theo lớp thô trên bản vẽ ISO. Code nằm ở plan Stage 2 (B6a, B6b).
2. ~~Mặc định `images: degraded` và báo cáo theo tờ gốc (2.2).~~ Đã chốt 2026-11-11: giữ `both`; tách sạch / suy giảm, số chính là suy giảm, báo cáo theo 12 tờ gốc (plan B6b, P4).
3. ~~(Tuỳ chọn) Kiểm chéo ảnh suy giảm (2.5); đưa `open_disc` vào `html` (2.4).~~ Đã chốt 2026-11-11: kiểm chéo ảnh suy giảm **bắt buộc trước R4** (plan B6d); `open_disc` hoãn sang Phase C (2.4).
4. (Mới, 2026-11-11) Vùng bỏ qua `iscrowd` cho mũi tên dẫn và cờ, trước R4 (2.6, plan B6d).
5. (Mới, 2026-11-11) Ghi chú khi viết luận văn: 2.3, 2.7 (plan Stage 2, cuối B8).
