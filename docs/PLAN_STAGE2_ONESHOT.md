# PLAN — Stage 2 one-shot: từ code GĐ2 đến kết quả luận văn (lập 2026-10-14)

Plan để **thực thi ở session sau**. Đọc kèm `docs/HANDOFF.md` (bối cảnh, quyết định đã chốt,
những gì đã thất bại). File này chỉ nói **làm gì, theo thứ tự nào, xong thì kiểm bằng gì**.

Ký hiệu: 🧑 = việc của user (Colab / chuyên môn P&ID) · 🤖 = việc code · ⛔ = chặn bước sau.

Nếu hướng này thất bại: xem `docs/BACKUP_PLAN.md` (tiêu chí thất bại, chẩn đoán, hướng thay thế).

---

## Quyết định 2026-11-09 (đọc trước; thay mọi chỗ mâu thuẫn bên dưới)

1. **Khung luận văn = phương án B.** *(Trọng số đổi 2026-11-12 → 40/40/20, xem "Quyết định
   2026-11-12 — trọng số" bên dưới; phần còn lại của mục này giữ nguyên.)* Tên đề tài ("Symbol detection and fine-grained
   classification in P&ID") không có "one-shot", nên:
   - **Lõi (~70%) — closed-set trên DigitizePID 32 lớp**: Stage 1 P/R/mAP; Stage 2 top-1,
     macro-F1, confusion; **mAP end-to-end so với YOLO một-stage 32 lớp fine**
     (`detector.label_level=fine`, đã có). Encoder R1b. B6c ✅ 2026-11-13.
   - **Chuyển sang ISO (~20%)**: OPEN100-ISO (render V8) chỉ làm **test**; nhãn ISO từ
     prototype của template. Protocol P4.
   - **Mở rộng (~10%)**: lớp ISO chưa thấy (P1 + H tự nhiên) và cổng "Other".
   - Hệ quả: **R1b chạy trước R1a**; "Other" từ đóng góp chính xuống mục mở rộng.
2. **Template ISO chỉ dùng lúc suy luận.** Encoder chỉ học từ crop DigitizePID. Không train
   gì trên template ISO, cũng không trên crop OPEN100-ISO.
3. **B7 (GĐ3, fine-tune với template ISO) BỎ** vì vi phạm mục 2. Thay bằng **B9 = cách (d)**:
   prototype = trung bình vector của template sạch và vài bản có đoạn ống ở cổng nối. Không
   train, nên "thêm ký hiệu = thêm 1 ảnh" vẫn đúng. **Hoãn (2026-11-10)**: B9 làm sau R4, khi
   đã có số của prototype sạch; không nằm trên đường chính.
4. **S/H nói trên lớp ISO, nhưng do dữ liệu train quyết định**, không chọn tự do: lớp ISO
   không có lớp DigitizePID tương ứng thì tự động là H ("H tự nhiên"). Xem mục "Lớp ISO thuộc
   nhóm nào" bên dưới. Lựa chọn duy nhất của ta là `h_in` / `h_out` trên DigitizePID (B5).
5. **Không train Stage 2 trên ISO-PID2Graph**: mọi mẫu của một lớp vẽ từ cùng một template
   (train/test vòng tròn), 12/42 lớp vocab không có mẫu nào, chỉ có 12 sơ đồ gốc.
6. Thuật ngữ trong luận văn: **deep metric learning (ArcFace) + so khớp prototype gần nhất**,
   không gọi là meta-learning / episodic training (ta không train theo episode).
7. **Stage 1 trên OPEN100-ISO chấm không phân lớp** (2026-11-11, V8_REVIEW §2.1): detector
   DigitizePID chỉ cần khoanh "ký hiệu"; nhãn ISO do Stage 2 gán. Trên bản vẽ ISO **không lọc
   prototype theo lớp thô** của Stage 1 (B6a). Bảng ánh xạ 8 lớp thô chỉ là báo cáo phụ.
8. **Train Stage 2 (R1b, R1a) với augment** (2026-11-11): `++fgc.aug.degrade.p=0.5` (suy giảm
   kiểu scan, vì đề thi chính là ảnh suy giảm) và `++fgc.aug.box_jitter.p=0.5` (B1c). Không làm
   ablation 2×2. Notebook đặt một chuỗi `AUG` dùng cho cả train lẫn resume.
9. **R1b chạy song song với việc code** (2026-11-11), ngoại lệ của quy tắc "code hết rồi mới
   chạy" (2026-10-14). Lý do: mọi việc code còn lại (B5b, B6a–B6e) thuộc phía suy luận / chấm,
   không đổi cách encoder được train. Điều kiện trước R1b: **H đã chốt** (chọn H sau khi thấy
   số R1b là cherry-picking), notebook có `AUG`, crop cắt theo định dạng mới (B1c). ✅ cả ba
   (2026-11-11). Cần push nhánh trước khi chạy (Colab clone từ GitHub). Hướng dẫn:
   `docs/COLAB_RUNBOOK.md` §5c.
10. **Stage 1 giữ detector coarse 8 lớp đã train**; thêm `agnostic_merge` lúc suy luận (B6e).
    Detector một lớp (`label_level=agnostic`) là tuỳ chọn để thử giả thuyết OLN (Kim et al.
    2022: detector không học phân lớp tìm vật thể chưa thấy tốt hơn) → bảng ablation Stage 1
    ba mức: 1 / 8 / 32 lớp.

Minh hoạ (sơ đồ): artifact "Hai hướng Stage 2" — https://claude.ai/artifact/JWSGBLAp2XTxufZru5BcuU

### Quyết định 2026-11-12 — trọng số luận văn 70/20/10 → 40/40/20

| Phần | Trước | Nay | Đề thi |
|---|---|---|---|
| Lõi: closed-set DigitizePID 32 lớp, hai-stage vs YOLO một-stage | 70% | **40%** | C |
| Chuyển sang ISO: OPEN100-ISO, nhãn ISO từ prototype | 20% | **40%** | P4 (S/V/H tự nhiên), P2 |
| Mở rộng: lớp chưa thấy + cổng "Other" | 10% | **20%** | P1, P4 (H chọn), AUROC/OSCR |

**Lý do.** R1b cho val_acc 0.9999 từ epoch 3 và Stage 1 coarse mAP50 0.995: DigitizePID đã bão
hoà, đề thi C gần như không còn khoảng phân biệt. Các kết quả có nội dung (separation, cụm van
"đặc điểm không cộng dồn", nhóm V/H) nằm ở phần chuyển sang ISO và mở rộng.

**Hệ quả.**
- C vẫn giữ (tên đề tài là detection + fine-grained classification), nhưng báo cáo trung thực là
  đã bão hoà; thêm mAP50:95 và, nếu kịp, C trên ảnh DigitizePID suy giảm (`scan_degrade`, seed
  cố định). B6c được phép dùng phương án rẻ (head per-group) nếu thiếu thời gian, ghi rõ.
- **Đường găng chuyển sang P4**: B6a (one-shot inference), B6b (P2/P4 + AUROC/OSCR + đường quét τ),
  B6d (đề thi OPEN100-ISO). Đường quét τ trên P4 là hình bắt buộc của chương chính.
- **R1a bắt buộc** (P1 nay là 20%), chạy sau P2 thu nhỏ. Chia B (đã đăng ký trước) đáng chạy nếu còn
  quota: với 20% trọng số, độ lệch giữa hai cách chia là câu hội đồng sẽ hỏi.
- **Phân tích lỗi chuyển giao thành một mục chính**: separation S/V/H, hai cụm, bảng gate/general/
  globe/butterfly (đặc điểm không cộng dồn), đối chiếu với nhóm V/H trên P4.
- **B4a (encoder chỉ-ImageNet)** tăng giá trị: cho biết train DigitizePID làm chuyển giao tốt lên
  hay tệ đi, đặc biệt ở cụm van. Nên làm nếu rẻ.
- **B9 (d)** vẫn sau R4, nhưng là ablation thuộc phần chính (40%), không còn là phần phụ.
- Lề template (Vấn đề mở 1) ảnh hưởng trực tiếp P4 → quyết bằng P2 thu nhỏ trước R4.

### Quyết định 2026-11-12 — tách val thành `val_sel` / `test`

**Bối cảnh.** DigitizePID chỉ có train/val, chia theo ảnh 400/100
(`digitizepid_splits.json`, ánh xạ id ảnh → "train"/"val"). Đã kiểm: không ảnh nào
có crop ở cả hai bên (id ảnh lấy từ tiền tố tên crop `<id ảnh>_<thứ tự>.png`).
Không có tập test, nên 100 ảnh val đang phải gánh cả chọn mô hình lẫn thi.

**Quyết định.** Tách 100 ảnh val thành hai nửa theo ảnh, không đụng 400 ảnh train:

| Tập | Số ảnh | Dùng cho |
|---|---|---|
| `val_sel` | 50 | chọn checkpoint, chọn τ (“Other”), chọn siêu tham số (d) |
| `test` | 50 | chỉ đề thi C, P1, P2; chạm **một lần**, khi checkpoint, τ, (d) đã chốt |

- Seed 20261112; file `digitizepid_val_test.json` (`{"val_sel": [...], "test": [...]}`).
- sha256: `digitizepid_splits.json` = `4dc786cf3f0d7eda7278f4f6a7e3816384a1988fe4c69729eab7d5a94c223d73`; `digitizepid_val_test.json` = `6a9f226487e6022f4cac7144235d13167f26fce98b1a851f76cc63fb62f75d3d`.
- Kiểm cân bằng: mọi lớp ≥142 crop ở mỗi nửa (Symbol_21 ~550). Không cần đổi seed.
- OPEN100-ISO không đổi vai trò: chỉ thi (P4), không chọn gì trên đó.

**Ghi nhận về R1b epoch 3.** Checkpoint `__global_ep3__.pt` được chọn theo `val_acc`
trên cả 100 ảnh val (trước khi tách). `val_acc` đã 0.9997 từ epoch 1 nên việc chọn này
gần như không mang thông tin; chấp nhận dùng epoch 3 để chẩn đoán trên `val_sel`. Nếu
cuối cùng dùng nó làm encoder chính thức, ghi rõ điều này khi báo cáo.

**Việc kéo theo (chưa làm).**
- Commit hai file split vào repo (vị trí: `configs/splits/` hoặc ngoại lệ `.gitignore`
  cho `data/canonical/`); notebook đọc split từ repo thay vì từ Drive.
- ~~`fgc_train`: tính `val_acc` chỉ trên `val_sel`; lưu `last.pt` mỗi epoch và checkpoint
  mốc 10/20/40/60.~~ → thay bằng quyết định 2026-11-13 bên dưới (epoch cố định; `val_acc` không
  còn dùng để chọn nên không cần tính trên `val_sel`).
- Đánh giá (B6b): thêm khóa config trỏ tới file; P2/P1/C đọc `test`, các bước chọn đọc
  `val_sel`. Không bước nào được đọc `test` trước khi chốt.

### Quyết định 2026-11-13 — chọn checkpoint theo epoch cố định (chốt TRƯỚC khi train R1a)

**Vấn đề.** `val_acc` bão hoà từ epoch 1 (0.9997 → 0.9999), nên "lưu khi `val_acc` tăng" chọn
epoch gần như ngẫu nhiên, và R1a (19 lớp) sẽ bão hoà ở epoch khác R1b. P1 so cùng một lớp giữa R1b
(đã thấy) và R1a (chưa thấy), nên hai encoder phải được lấy theo cùng một quy tắc, chốt trước khi
có số (như H).

**Quyết định (user đồng ý 2026-11-13).**
- R1a và R1b so ở **cùng epoch 3** (`CMP_EPOCH` trong notebook). R1b đã có epoch 3
  (`encoders/__global__.pt`), không train lại. R1a train `fgc.epochs=5`, gọn trong một phiên.
- Không resume R1b (vấn đề mở 3 → đóng): thêm epoch không sửa được lỗi chuyển giao, vì đó là giới
  hạn dữ liệu ("đặc điểm không cộng dồn").
- Không chọn checkpoint bằng `val_acc`, cũng không bằng P2 thu nhỏ (chọn theo P2 rồi báo P2 là
  dùng cùng một dữ liệu hai lần). Epoch 5 chỉ để kiểm độ nhạy: báo kèm nếu kết luận đổi giữa 3 và 5.
- Lượt train sau (chia B, nhãn theo nghĩa ISO) dùng cùng quy tắc.

**Code (2026-11-13).**
- `fgc_train`: ghi `<group>.last.pt` mỗi epoch (có `best_val_acc`); ghi `<group>.ep<N>.pt` ở mỗi
  epoch trong `++fgc.save_epochs` (key mới qua `++`, chưa vào `config.yaml`; mặc định không ghi),
  bất kể `val_acc`; cảnh báo nếu N > `fgc.epochs`. `<group>.pt` (best `val_acc`) giữ như cũ.
  `_maybe_resume` ưu tiên `.last.pt`, không có thì `<group>.pt` (run cũ).
- Notebook: `CMP_EPOCH = 3`, `EPOCHS = 5`, `SAVE = '++fgc.save_epochs=[3,5]'` ở mục 4, cho cả 4
  lệnh train. `finalize` tách sang cell riêng (4b, chỉ định nghĩa), nhận tên run (`'r1a'`/`'r1b'`),
  tìm checkpoint epoch `CMP_EPOCH` trong mọi phiên của run (`.ep3.pt`, hoặc `__global__.pt` của
  run cũ nếu đúng epoch 3). Sửa luôn bẫy `IndexError`/`NameError` khi R1b chạy trước R1a.
- Test: `test_exclude_classes.py` (+2), `test_resume.py` (+1), `test_colab_notebook.py` (+2).

---

### Vấn đề mở 2026-11-12 (từ R1b + R2, chờ quyết)

1. **Lề template ≠ lề crop.** `iso_prototypes` nhúng template sát bbox vector; mọi crop từ B1c là
   box + 10% mỗi bên (`CROP_MARGIN_FRAC`). Ứng viên: `oneshot.template_views.scales=[1.2]`
   (`iso_prototypes_m12.pt`). Separation không đổi (0.240 / 0.990) → quyết bằng P2 thu nhỏ trên
   `val_sel`, so hai bản prototype. Đây là câu hỏi nhất quán hình học, không phải ablation; nếu
   chọn 1.2 thì sửa quyết định thiết kế GĐ2 số 1 (HANDOFF §2) và ghi lý do.
   **P2 thu nhỏ (2026-11-12): không bản nào thắng hẳn** — macro S 0.70 (1.0) vs 0.67 (1.2); 1.2
   sửa một phần S3 (0% → 66%) nhưng làm hỏng S10 (own_cos 0.68 → 0.22). Prototype một ảnh rất
   nhạy với cách đóng khung → thử tiếp `template_views.scales=[1.0, 1.2]` (trung bình, cơ chế có
   sẵn) trước khi chốt.
2. ~~**Đề thi C đã bão hoà**~~ → **đã quyết 2026-11-12**: trọng số 40/40/20 (mục "Quyết định
   2026-11-12 — trọng số"). Còn mở: có làm C trên ảnh DigitizePID suy giảm hay không.
3. ~~**Có resume R1b hay không**~~ → **đã quyết 2026-11-13**: không resume; R1a và R1b so ở
   epoch 3 (mục "Quyết định 2026-11-13 — chọn checkpoint").
4. **★ `pipeline` hút crop có ống.** P2 thu nhỏ: 100% crop S24 (flow_direction) ra `pipeline`
   (cos 0.88 > own 0.83 ở scale 1.0; 0.965 > 0.637 ở 1.2). Mọi crop mũi tên đều kèm đoạn ống, ★
   `pipeline` chỉ là một đường thẳng. 🧑 Quyết: nếu Stage 1 không bao giờ khoanh "đường ống" như một
   ký hiệu thì bỏ `pipeline` (và các template thuần đường kẻ tương tự) khỏi thư mục prototype; ảnh
   hưởng cả P4.
5. **Trạng thái của S21 (flanged_connection).** P2 thu nhỏ: own_cos chỉ 0.27–0.29 (ngang lớp
   VARIANT), 100% ra `agitator_general`. 🧑 Xem crop S21 cạnh template mặt bích (như đã làm với S6):
   có thật là cùng hình vẽ không, hay nên là VARIANT. Đổi trạng thái → chạy lại `check_b5.py`, đếm
   lại S/V/H.
6. **ArcFace đẩy VARIANT ra xa ★ của chính nó.** P2 thu nhỏ: 8/9 lớp VARIANT sai, own_cos
   0.07–0.25 (chỉ S22 hx_coil đúng, 0.95). Với S2, S13 (van bi) và S12 (van chung): R1b train
   chúng là lớp RIÊNG với S5 / S4, tức chủ động tách hai cách vẽ của cùng một ký hiệu — mâu thuẫn
   trực tiếp với kỳ vọng "VARIANT khớp ★ ISO". Các VARIANT không có lớp anh em (S6, S8, S14, S20,
   S23) đơn giản là vẽ khác xa template. Một số lỗi rất tự tin (S8 → reducer 0.93, S23 →
   hx_straight_tube 0.83) → τ không chặn được. Lựa chọn (chưa làm): train bằng nhãn theo nghĩa ISO
   (gộp S2, S13 → S5; S12 → S4) — đổi lại C không còn đúng 32 lớp. Là kết quả chính của phần chuyển
   giao (40%), ghi vào luận văn dù chọn hướng nào.

#### Kết quả P2 thu nhỏ (2026-11-12) — R1b epoch 3, `val_sel`, 50 crop/lớp, không τ
Script `p2_mini.py` (đúng đường code repo: `view` box+10% → `query_views` 8 view →
`build_eval_transform` → `match_views`). CSV: `runs/p2_mini_q50/`.

| | scale 1.0 | scale 1.2 |
|---|---|---|
| Macro S (10 lớp; vote / mean / best) | 0.70 / 0.70 / 0.70 | 0.68 / 0.70 / 0.67 |
| Macro V (9 lớp VARIANT) | 0.11 | 0.22 (S13 đúng do may, own 0.25) |
| S3 → valve_butterfly (best) | 100% | 34% |
| Lớp S sai | S3, S21, S24 | S10, S21, S24 (S3 đúng 66%) |

- **Mỗi lớp ≈ một điểm dữ liệu**: trong một lớp, cả 50 crop cùng một dự đoán, own_cos dao động
  < 0.03 (DigitizePID tổng hợp, cùng một glyph). P2 trên DigitizePID thực chất đo "glyph
  DigitizePID có gần template ISO không" với ~10 điểm (S). Báo cáo THEO LỚP; không trình bày như
  độ chính xác trên hàng nghìn crop. Chạy đủ crop không đổi kết luận.
- vote ≈ mean ≈ best → view xoay không gây nhiễu; luật bỏ phiếu không phải vấn đề lúc này.
- Ba lỗi S có ba nguyên nhân: S3 (cụm van, "đặc điểm không cộng dồn"), S24 (★ `pipeline`, vấn
  đề mở 4), S21 (glyph khác template, vấn đề mở 5).
- Gợi ý cho τ: lớp S đúng có own_cos 0.54–0.99; VARIANT hầu hết < 0.3 nhưng có lỗi tự tin tới 0.93.

## 0. Trạng thái xuất phát

| Hạng mục | Trạng thái |
|---|---|
| Stage 1 (YOLOv8, 8 coarse) | ✅ mAP50=0.995 — **không train lại**. `best.pt` nằm trên Drive, không có ở local |
| GĐ2 one-shot core (`utils/oneshot.py`, `pid-iso-prototypes`) | ✅ code xong, ✅ pytest xanh (B0, 2026-10-18) |
| `fgc.global_encoder` (prep-crops / train / infer guard) | ✅ code xong, ✅ pytest xanh (B0, 2026-10-18) |
| Per-subpart augment (OSSR-PID #2) | ✅ `SubpartWarp` (B1, 2026-10-18), user đã duyệt ảnh |
| GĐ4 one-shot inference | ✅ B6a 2026-11-13 (`pipeline/oneshot_infer.py`, `fgc.oneshot=true`) |
| GĐ5 eval one-shot + "Other" | ❌ chưa code |
| Leave-classes-out (`fgc.exclude_classes`) — protocol chính để chứng minh one-shot | ✅ code + test (B1b, 2026-10-18); danh sách H vẫn chờ B5 |
| `data/canonical/` local | ✅ DigitizePID gốc: 500 ảnh (400 train / 100 val), 59,498 ann, 32 class `Symbol_N` |
| `data/derived/` local | chỉ có `yolo_coarse/` — chưa có `crops_fgc/`, `tiles/` |
| Crop DigitizePID trên Drive (`crops_fgc.zip`) | ⚠️ **chỉ có 28/32 class** — phải cắt lại |
| `symbol_map.csv` (2026-11-09) | ✅ 32/32 có status: **10 ISO / 11 OOV / 9 VARIANT / 2 UNSURE**; `check_b5.py` 0 lỗi, chỉ cảnh báo H rỗng |
| OPEN100-ISO (render V8) | ✅ 864 ảnh (432 sạch + 432 suy giảm), 93,800 annotation; **30/42** lớp vocab có mẫu (`docs/experiments/iso_render/v8/`) |
| Augment suy giảm kiểu scan (`+fgc.aug.degrade.p`, `+detector.aug.degrade.p`) | ✅ code + test, mặc định tắt (`src/utils/scan_degrade.py`) |
| Closed-set infer / eval với `__global__.pt` | ✅ B6c 2026-11-13: `fgc_infer` dùng `__global__.pt`, `evaluate eval.mode=fgc_only`, YOLO fine chấm bằng `det_only detector.label_level=fine` |
| Luật VARIANT trong `check_b5.py` | ✅ B5b 2026-11-13; `holdout_classes` trả `(h_in, h_out, h_extra)` |
| Crop có ngữ cảnh + box jitter (B1c) | ✅ 2026-11-11 (`utils/crop_box.py`); sửa luôn lệch train/suy luận: `fgc_infer` trước cắt box trần, train có lề 10% |
---

## Protocol đánh giá — chứng minh one-shot ĐÚNG NGHĨA (user chốt 2026-10-14)

**Vấn đề**: nếu encoder được train trên cả 32 class rồi test bằng crop của chính 32 class đó
(chỉ đổi nhãn sang ISO), accuracy cao có 2 cách giải thích mà phép test KHÔNG phân biệt được:
(A) encoder học đặc trưng tổng quát nên 1 template là đủ — điều ta tuyên bố; (B) encoder đã
thấy ~1,700 mẫu của class đó khi train, template chỉ rơi vào cụm có sẵn — tức "1 template +
1,700 mẫu", không phải one-shot. Cùng lỗi với "Other": class OOV đã có trong tập train nên
reject chúng dễ hơn một symbol thật sự lạ. Và số của OSSR-PID (1 template/class) không còn
so được với số của ta.

**Cách vá — leave-classes-out**: giữ hẳn một số class ra khỏi train encoder, test trên chính
chúng. Encoder chưa từng thấy mẫu nào → nhận đúng chỉ có thể nhờ template (giải thích A).

| Protocol | Encoder | Tập test | Đo gì | Vai trò |
|---|---|---|---|---|
| **C — closed-set** (2026-11-09) | `__global__.pt` (đủ 32 class), head softmax | crop + bản vẽ DigitizePID `test` (50 ảnh, 2026-11-12) | top-1, macro-F1, confusion 32 lớp; mAP end-to-end so với YOLO 32 lớp fine | **Lõi luận văn** (phương án B) |
| **P1 — unseen-class one-shot** | `__global_lo__.pt` (train trên 32 − H class) | crop của H_in (giữ ra, map được ISO) + H_out (giữ ra, OOV) | one-shot acc trên H_in; "Other" với dương = H_out, âm = H_in (**cả hai đều unseen**) | Bằng chứng chính của mục mở rộng: lớp chưa thấy, **nét vẽ khác template** |
| P2 — seen-class alignment | `__global__.pt` (đủ 32 class) | crop các class map được ISO | template ISO (nét vector) có khớp crop raster không | Kết quả phụ, ghi rõ KHÔNG phải one-shot; P2 − P1 = cái giá của "unseen" |
| P3 — real, unseen (nếu có) | `__global__.pt` | ASU PID_Dataset (`~/Downloads/PID_Dataset.zip`, Stage-2 test: 12 sheet thật, ~102 class) | như P1 trên bản vẽ thật | Bằng chứng mạnh nhất; chờ user quyết định giải nén + map class |
| **P4 — OPEN100-ISO** (2026-11-09) | cả hai encoder | crop GT + bản vẽ OPEN100-ISO (sạch / suy giảm) | acc theo nhóm **S / V / H tự nhiên / H chọn**; "Other" với dương = `other_iso`; end-to-end; theo mức suy giảm | Phần "chuyển sang ISO". H ở đây **cùng nét vẽ** với template nên dễ hơn P1 — báo cáo tách riêng, không gộp với P1 |
| Baseline | encoder chỉ-ImageNet (B4a) | như P1 | encoder chưa học P&ID | Cho thấy train DigitizePID đóng góp bao nhiêu |

Quy tắc bắt buộc:
- **Danh sách H chọn TRƯỚC khi train và TRƯỚC khi thấy bất kỳ kết quả nào** (B5), ghi cố định
  vào config + HANDOFF — chọn sau khi thấy số là cherry-picking.
- **Ngưỡng "Other" (τ) không bao giờ nhìn lớp H hay OPEN100-ISO** (thay 2026-11-11; trước:
  calibrate trên crop H của split `train` — dùng mẫu "lạ" để chỉnh ngưỡng, số lạc quan). τ chỉ
  chọn trên `val_sel` DigitizePID (50 ảnh, quyết định 2026-11-12), crop của các lớp **đã thấy**, theo quy
  tắc cố định — xem B8 "Ngưỡng Other". So sánh giữa các phiên bản bằng AUROC / OSCR (không phụ
  thuộc ngưỡng).
- **Chặn rò rỉ bằng code** (B1b, B6b): eval P1 phải từ chối chạy nếu một class test nằm trong
  `fine_classes` của checkpoint.
- So với OSSR-PID **chỉ bằng số P1**, kèm câu nêu rõ khác biệt protocol.

### Lớp ISO thuộc nhóm nào (S / V / H) — 2026-11-09

Một lớp ISO là "đã thấy" với encoder E khi E đã train trên lớp DigitizePID tương ứng. Nhóm được
**tính tự động** từ `symbol_map.csv` (cột `status`, `reg_number`) và danh sách lớp lưu trong
checkpoint (`fine_classes`, `excluded_classes`) — không gõ tay (B6b):

| Nhóm | Điều kiện (với encoder E) | Với R1b (đủ 32 lớp) |
|---|---|---|
| **S** — đã thấy | có lớp DigitizePID `status=ISO` cùng `reg_number`, nằm trong `fine_classes` của E | 10 lớp: `valve_gate/globe/general/ball/needle`, `open_disc`, `interchangeable_disc(_open)`, `flanged_connection`, `flow_direction` |
| **V** — biến thể | không thuộc S; có lớp `status=VARIANT` cùng `reg_number` trong `fine_classes` của E | `valve_butterfly`, `valve_control`, `check_valve_general`, `reducer`, `hx_coil`, `piping_heated_cooled_insulated` |
| **H chọn** | lớp `status=ISO` tương ứng nằm trong `excluded_classes` của E | rỗng (chỉ có ở R1a) |
| **H tự nhiên** | không lớp DigitizePID nào map tới | 26 lớp còn lại của vocab 42 |
| **Other** | ký hiệu không có trong vocab 42 | DigitizePID: 11 lớp OOV. OPEN100-ISO: `other_iso` (19 template render ngoài vocab, 10,144 instance) |

Trên OPEN100-ISO chỉ 30/42 lớp có mẫu: S 9 lớp (không có `open_disc`), V 4 lớp, **H tự nhiên 17
lớp** (8,444 instance tính cả bản sạch và suy giảm: bơm, máy nén, tháp, agitator,
`check_valve_globe/swing`, `safety_valve_spring`, …). 11 lớp vocab không có template render (`tank`, `motor_electric`,
`hx_*`, `open_disc`, …) vẫn có prototype và vẫn là ứng viên khi so khớp (gây nhầm), chỉ không có
mẫu test.

`other_isa` (bong bóng ISA trên OPEN100-ISO) là "Other" **đã thấy** (S26–S32 có trong train
R1b) → báo cáo tách khỏi `other_iso`.

---

## Thứ tự: CODE HẾT TRƯỚC, RỒI MỚI CHẠY (user chốt 2026-10-14)

**Phase A — Code** (🤖, local, test bằng encoder giả / model chưa train; không cần Colab):

```
(B0 ✅) ─► (B1 ✅) ─► (B1b ✅) ─► (B2 ✅) ─► (B1c ✅) ─► (B5b ✅) ─► (B6c ✅) ─► (B6a ✅) ─► B6b GĐ5
  ─► B6d đề thi OPEN100-ISO ─► B6e Stage 1 agnostic_merge ─► B4a baseline
B5 🧑 CHỌN danh sách H (+ B5b 🤖 luật VARIANT trong check_b5) ── song song, ⛔ PHẢI XONG trước R1 ──►
```

**Phase B — Chạy từng GĐ** (🧑 Colab, theo thứ tự, mỗi bước xong mới sang bước sau):

```
H chốt ─► R1b = train encoder ĐỦ 32 class        (song song với Phase A, quyết định 9)
       ─► R1a = train encoder LEAVE-OUT          (sau B5b)
       ─► YOLO 32 lớp fine                       (baseline một-stage cho C)
       ─► (tuỳ chọn) YOLO một lớp                (ablation Stage 1, B6e)
  ─► R2 = prototypes (sạch) + separation (cả hai encoder)      (sau Phase A)
  ─► R4 = B8 eval C (lõi) + P1/P2/P4 (+ P3) + τ trên val DigitizePID + chốt số
```

**Phase C — Sau R4, tùy chọn (hoãn 2026-11-10)**: B9 (d) code ─► R2' prototype có stub (không
train lại, dùng encoder R1a/R1b có sẵn) ─► R4' chạy lại P1 + P4 với cùng ngưỡng → một dòng
ablation. Không cần Colab train nên làm muộn không tốn quota.

R1b trước R1a (đổi 2026-11-09): R1b phục vụ phần lõi closed-set, P2 và P4; R1a chỉ phục vụ mục
mở rộng. Danh sách H vẫn phải chốt **trước R1b**, vì chọn H sau khi thấy số của R1b là
cherry-picking. Crop chỉ cần cắt MỘT lần (`global_encoder=true` cắt đủ 32 class, kể cả class H
— cần chúng để test). R3 (GĐ3) không còn: B7 đã bỏ.

Vì sao thứ tự này ổn: không bước code nào cần checkpoint thật — mọi code đều test được bằng
`_FakeEncoder` / ArcFaceModel chưa train (như `tests/test_oneshot.py` đã làm). B1 tự nhiên
nằm trước R1 nên không có nguy cơ train rồi mới thêm augment.

**Cái giá phải chấp nhận**:
- Code B6a/B6b chỉ được kiểm với encoder giả → lỗi ghép với model thật lộ ra ở R2/R4. Giảm
  bằng cách test end-to-end với ArcFaceModel chưa train trên crop thật (như
  `test_iso_prototypes.py` làm với 39 symbol ISO thật).
- Câu hỏi khoa học lớn nhất — encoder DigitizePID có chuyển sang ISO không — chỉ trả lời ở R2.
  Nếu R2 kém, thử B9 (d) trước (Phase C, không train lại); vẫn kém thì không còn GĐ3 để cứu
  (B7 đã bỏ) → `docs/BACKUP_PLAN.md`.
- **B4a** (tùy chọn, đề xuất — user chưa chốt): encoder chỉ-ImageNet làm baseline. Chạy được
  ngay đầu Phase B, trước cả R1, cho tín hiệu separation sớm + một dòng ablation
  "ImageNet-only vs train DigitizePID".

---

## B0 — Verify code chưa từng chạy 🤖 ✅ XONG 2026-10-18

Chạy local trong venv của uv (`uv` 0.13.1 Homebrew, Python 3.11.17, torch 2.13.0 CPU/MPS):

```bash
uv sync --extra dev
uv run pytest tests/test_oneshot.py tests/test_iso_prototypes.py tests/test_global_encoder.py
uv run pytest
uv run ruff check src/ tests/ scripts/
```

Kết quả: 91/91 test mới pass, toàn bộ suite 207/207 pass. Ruff chỉ báo một lỗi I001 (dòng trống
thừa sau block import `importorskip` trong `tests/test_global_encoder.py`), đã sửa bằng
`ruff check --fix`. Không phải sửa code `src/`. Còn lại: warning `Image.getdata` deprecated
(Pillow 14, 2027-10) trong `tests/test_oneshot.py`, chưa ảnh hưởng.

---

## B1 — Per-subpart augment (OSSR-PID kỹ thuật #2) 🤖 ✅ XONG 2026-10-18

**Vì sao**: các valve DigitizePID khác nhau ở chi tiết nhỏ (chấm giữa, X chéo, actuator).
Augment cả crop không dạy được model rằng *từng bộ phận* có thể lệch nhẹ mà danh tính giữ
nguyên — OSSR-PID affine riêng từng subpart vì lý do này.

**Trạng thái**: ✅ XONG 2026-10-18 — code + test xanh (229 pass, ruff sạch) · 🧑 user đã duyệt ảnh.

**Thiết kế ban đầu bị bỏ** — affine riêng từng connected component (±20°), fallback affine
cả crop. Đo trên ~1,600 crop thật (12 ảnh): chỉ **0–13% crop valve** (Symbol_1–16; riêng
Symbol_5 37%) tách được ≥2 component — thân, ống, trang trí là MỘT nét liền. Với valve nó
thoái hoá thành xoay cả crop. Đã thử chia nét bằng k-means trên toạ độ mực → nét bị rách ở
chỗ cắt (artefact bản vẽ thật không có) → cũng bỏ.

**Thiết kế đã chốt — `SubpartWarp`** (user chọn "phương án 3" 2026-10-18), bản raster gần
đúng của kỹ thuật OSSR-PID, trên crop gốc **trước** `Resize`:
1. Otsu → mực; nền = median pixel không-mực (giữ nền xám của DigitizePID, không ép trắng);
   làm việc trong không gian "độ tối trên nền", ghép lại bằng mực-tối-thắng.
2. Component đủ lớn (≥3% lượng mực), nếu ≥2: affine **độc lập** quanh tâm — rotation ±8°,
   scale 0.92–1.08, shear ±0.05 (nhỏ hơn ±20° của OSSR-PID vì xoay cả symbol là việc của
   `fgc.aug.rotate`).
3. Trường dịch chuyển mượt (lưới 3×3 nội suy cubic, tối đa 4% cạnh crop): dịch các *vùng*
   của symbol so với nhau (tỉ lệ cone, cỡ chấm, lệch actuator) mà **không làm đứt nét**.
4. Độ dày nét ±1px, **lệch về mảnh** (p_thin 0.35 / p_thick 0.15): template ISO vẽ nét mảnh
   hơn crop DigitizePID ~1.1–2× (độ dày / cạnh symbol: ISO 0.025–0.037 vs crop 0.041–0.051).

**Bằng chứng (mini-P1; ảnh, số liệu, script: `docs/experiments/b1_subpart/`)**: ResNet34+ArcFace train 12 class valve
(250 crop/class, 6 epoch), giữ ra S1/S3/S4/S10, nhận bằng template ISO thật
(X8074/X8068/2101/X8076 — map lấy từ cột `reg_number` của `symbol_map.csv`, chưa chốt B5),
4 view xoay, 408 crop val, 5 seed:

| Cấu hình | template ISO | 1 crop DigitizePID (đối chứng) |
|---|---|---|
| chỉ ImageNet, không train | 0.706 | 0.613 |
| không subpart | 0.691 ± 0.151 | 0.825 ± 0.054 |
| CC-only (thiết kế ban đầu) | 0.670 ± 0.105 | 0.909 ± 0.108 |
| **`SubpartWarp`** | **0.770 ± 0.201** | **0.906 ± 0.073** |

Đọc: CC-only giúp trong miền nhưng KHÔNG giúp sang ISO (−0.021 so với không subpart); train
không augment còn thua baseline ImageNet trên ISO → động lực cho B1. `SubpartWarp` tốt nhất
trung bình nhưng **độ lệch seed ngang khoảng chênh** (4 class test, kết quả từng class phân
cực 0/1) → là *hướng*, chưa phải kết quả; ablation `fgc.aug.subpart.p=0` vs bật ở R1/R4 mới
quyết định.

**Code**: `src/utils/augment.py` (`SubpartWarp`, PIL→PIL, `seed=None` dùng `random` toàn cục
để DataLoader worker reseed được) · `src/utils/dataset.py::_build_transform` (chèn trước
`Resize`, chỉ train, bỏ hẳn khi `p=0`) · `configs/config.yaml` `fgc.aug.subpart` ·
`tests/test_augment.py` (22 test: p=0 y hệt, nét liền vẫn liền, part rời vẫn rời, nền giữ
xám, mực không mất, thin/thick đúng chiều, eval transform không augment).

**Xong khi**: 🧑 duyệt `docs/experiments/b1_subpart/B1_review.png` (S1/3/7/10/13/14) và
`B1_review_pairs.png` (cặp dễ nhầm S12↔S13, S15↔S16, S2↔S5) — augment không được làm mất
chi tiết phân biệt class.

**Phát hiện phụ (đã xử lý 2026-10-18)**: template `X8071.png` bị lỗi → cả bộ
`data/raw/iso_symbols_clean/` được tạo lại từ SVG vector BK-2020-04 (`scripts/extract_iso_symbols_svg.py`,
chi tiết `docs/experiments/iso_templates_bk2020/`). Mini-P1 ở trên đo trên bộ template CŨ.

---

## B1b — Leave-classes-out: `fgc.exclude_classes` 🤖 ✅ XONG 2026-10-18

**Đã làm** (246 test pass, ruff sạch; override Hydra kiểm trên `classes.yaml` thật — giữ ra 2
class → head global 30 class; tên sai → lỗi):
- `utils/classmap.py::training_fine_classes(classmap, group, exclude)` — điểm quyết định DUY
  NHẤT; dùng bởi `fgc_train`, `dataset.build_dataset`, `fgc_infer`, `evaluation` (cả 4 chỗ
  dựng head/label). Tên lạ → `ValueError`; giữ ra hết class của một group → `ValueError`.
- `utils/dataset.py::excluded_classes(cfg)`; `configs/config.yaml` `fgc.exclude_classes: []`.
- Checkpoint lưu `excluded_classes`; `pid-iso-prototypes` ghi `encoder_excluded_classes` vào
  `meta` (checkpoint cũ không có key → `[]`, `utils/model.py::checkpoint_excluded_classes`).
- **Thêm ngoài plan**: resume từ checkpoint train trên danh sách class KHÁC → `ValueError`
  (cùng số class giữ ra nhưng khác tên thì head cùng shape, `load_state_dict` không báo lỗi
  → cột lệch ngầm; dễ xảy ra khi gõ lại override ở phiên Colab sau).
- `tests/test_exclude_classes.py` (17 test): rỗng = hành vi cũ; dataset train/val không có
  crop class H; label khớp `fine_classes` checkpoint; head 6−K hàng; typo dừng trước khi
  train; infer dựng head per-group đúng chiều rộng; resume chặn danh sách khác.
- `extract_crops` KHÔNG đổi — crop class H vẫn được cắt (cần cho test P1).

**Kế hoạch gốc** (giữ để đối chiếu):

**Vì sao**: xem mục "Protocol đánh giá" — không có bước này thì không chứng minh được one-shot.

**Code**:
- `configs/config.yaml`: `fgc.exclude_classes: []` — danh sách fine class (`Symbol_N`) bị
  giữ ra khỏi TRAIN. Comment ghi rõ: điền đúng danh sách H đã chốt ở B5, không sửa sau khi
  đã thấy kết quả.
- Một hàm duy nhất quyết định "class nào được train", ví dụ
  `utils/classmap.py::training_fine_classes(classmap, group, exclude)` — dùng chung cho
  `fgc_train._train_group` VÀ `utils/dataset.build_dataset`, để thứ tự cột head và label của
  dataset không bao giờ lệch nhau.
- **KHÔNG** đổi `extract_crops`: vẫn cắt crop của class H (cần cho test).
- Checkpoint lưu thêm `excluded_classes`; `pid-iso-prototypes` ghi nó vào `meta`.
- Validate: tên không có trong `classes.yaml` → lỗi; loại hết class → lỗi.
- Tests (`tests/test_exclude_classes.py`): head có 32 − K hàng; dataset không chứa crop nào
  của class H; label index khớp thứ tự `fine_classes` trong checkpoint; danh sách rỗng =
  hành vi cũ y nguyên; tên sai → lỗi rõ ràng.

---

## B1c — Crop có ngữ cảnh + box jitter 🤖 ✅ XONG 2026-11-11

**Vì sao**: box từ detector không trùng box GT (lệch tâm, rộng / hẹp hơn); crop train chỉ có lề
cố định 10% nên encoder chưa từng thấy crop lệch. Khi làm thì lộ thêm một lỗi: `fgc_infer` cắt
**box trần** trong khi crop train có lề 10% — train và suy luận nhìn hai view khác nhau.

**Đã làm** (`src/utils/crop_box.py`, 602 test pass, ruff sạch):
- Một quy tắc cho mọi nơi: view phân loại = box + `CROP_MARGIN_FRAC` (10%) mỗi cạnh, cắt theo
  mép ảnh (`cut`). `fgc_infer` dùng `cut` → hết lệch.
- `pid-prep-crops` lưu **vùng ngữ cảnh** (box + 40% mỗi cạnh) và `<lớp>/_boxes.csv` (box + gốc
  toạ độ). Dataset cắt view từ đó: val / test đúng từng pixel như `cut` lúc suy luận (test 300
  box ngẫu nhiên, kể cả box chạm mép ảnh); train cắt quanh box lệch (`BoxJitter`).
- `BoxJitter` (`++fgc.aug.box_jitter.p`, mặc định tắt): dịch tâm ≤ 8% cạnh box, co giãn
  0.9–1.15 (log-uniform), tỉ lệ khung 0.9–1.1; kiểm dải không vượt vùng ngữ cảnh đã lưu.
  IoU box lệch / GT > 0.6 trên 500 lần bốc. Ngữ cảnh thật (ống, chữ bên cạnh), không đệm nền.
- Crop định dạng cũ (không `_boxes.csv`): vẫn dùng được như cũ; bật jitter thì dừng với lỗi
  "re-run pid-prep-crops".
- Notebook: zip mới `crops_fgc_ctx.zip` (không bao giờ đọc `crops_fgc_global.zip` cũ), assert
  `_boxes.csv` ở mọi lớp, chuỗi `AUG` cho R1a/R1b cả train lẫn resume. `tests/test_colab_notebook.py`
  chặn quên `AUG` khi resume.
- Tests: `tests/test_crop_box.py` (9).
- Đã xem trên ảnh DigitizePID thật (6 lớp × 5 view lệch): ký hiệu luôn trọn trong khung.

---

## B2 — Colab plumbing 🤖 ✅ XONG 2026-10-19

**Đã làm** (281 test pass, ruff sạch):
- `run_pipeline.py prototypes` → `IsoPrototypePipeline` (`tests/test_run_pipeline.py`).
- **Danh sách H có một nguồn duy nhất**: `oneshot.holdout.{h_in,h_out}` trong config,
  `utils/oneshot.py::holdout_classes(cfg)` (kiểm trùng, kiểm giao nhau). Notebook dựng
  `fgc.exclude_classes` từ đây, KHÔNG gõ tay; B6b đọc cùng chỗ.
- Notebook `train_stage2_colab.ipynb` viết lại (25 cell): tự dò Drive (`MyDrive/pid` hoặc
  `MyDrive/PID_Recognition`, hai notebook cũ lệch nhau); cài đủ deps (bản cũ thiếu
  ultralytics/sahi/ensemble-boxes, mà `run_pipeline.py` import ngay đầu file); cắt crop
  toàn cục ở LOCAL → `crops_fgc_global.zip` + assert đủ 32 class (không bao giờ đụng
  `crops_fgc/` 28 class trên Drive: tên file đánh số khác nhau, cắt chồng sẽ ra crop
  trùng); R1a / R1b có cell resume (tự lấy run mới nhất); cell `finalize` kiểm rò rỉ
  (không cột nào của H, đúng 32 − K hàng, `excluded_classes` khớp) rồi copy thành
  `runs/encoders/__global_lo__.pt` / `__global__.pt`; R2 prototypes cho cả hai encoder
  (template + vocab lấy từ repo); cell eval comment lại.
- `tests/test_colab_notebook.py`: chặn các bẫy cũ quay lại (zip 28 class, train không
  global, H gõ tay, eval theo group).
- `docs/COLAB_RUNBOOK.md` §5b.
- **Chạy khô** đúng các lệnh notebook trên 7 ảnh thật (4 train / 3 val): crop đủ 32/32 class
  ở cả 2 split → R1a 29 class (H ví dụ 3 class) → resume cùng H ok, khác H bị chặn → R1b
  32 class → finalize đúng, phát hiện rò rỉ khi kỳ vọng sai → prototypes ×2, meta ghi
  `encoder_excluded_classes`.
- ~~Lưu ý (có sẵn từ trước, chưa sửa): checkpoint chỉ ghi khi val_acc tăng → resume có thể
  lặp lại epoch.~~ → sửa 2026-11-13: `.last.pt` mỗi epoch, `.ep<N>.pt` theo `save_epochs`.

**Kế hoạch gốc** (giữ để đối chiếu):

Notebook `notebooks/train_stage2_colab.ipynb` hiện gọi `src/run_pipeline.py`, **chưa** biết
global encoder, và sẽ **tái dùng `crops_fgc.zip` cũ trên Drive (28 class)** — bẫy im lặng.

1. `src/run_pipeline.py`: thêm subcommand `prototypes` → `IsoPrototypePipeline`
   (hiện chỉ có entrypoint `pid-iso-prototypes`, notebook không gọi được).
2. Notebook:
   - Cell giải nén crop: đổi tên zip thành `crops_fgc_global.zip` để **không bao giờ** nhặt
     nhầm zip 28 class. Thêm assert sau khi giải nén:
     `assert len(os.listdir(f'{LOCAL_CROPS}/train')) == 32`.
   - Cell cắt crop (nếu chưa có zip): `prep --step crops fgc.global_encoder=true`.
   - Cell train: thêm `fgc.global_encoder=true`.
   - Cell resume: giữ nguyên, chỉ thêm `fgc.global_encoder=true` (ckpt là `__global__.pt`).
   - Cell eval cũ: **comment lại** kèm ghi chú — `evaluation.py` lặp theo `fgc_groups`, sẽ
     bỏ qua im lặng với global encoder cho tới khi B6b xong.
   - Cell mới: `prototypes oneshot.encoder_ckpt=.../__global__.pt` + in log separation.
   - Cell train leave-out (R1a): như cell train nhưng thêm `fgc.exclude_classes=[...]`
     (copy đúng danh sách H từ config, không gõ tay lại) và `run_dir` riêng. Assert sau train:
     checkpoint có `32 − K` class và không chứa class nào của H.
3. Cập nhật `docs/COLAB_RUNBOOK.md` theo các cell mới.

**Xong khi**: notebook chạy khô được tới cell train (có thể kiểm bằng `fgc.epochs=1` trên
vài crop local nếu có data).

---

## B3 — Train encoder toàn cục 🧑 (Colab) — R1a leave-out, R1b đủ 32 class

```
prep --step crops   fgc.global_encoder=true                       # 1 lần, đủ 32 class, định dạng B1c
AUG="++fgc.aug.degrade.p=0.5 ++fgc.aug.box_jitter.p=0.5"         # quyết định 8; cả train lẫn resume
# R1b — encoder đủ 32 class: lõi closed-set (C), P2, P4, end-to-end (chạy TRƯỚC, 2026-11-09):
fgc --train         fgc.global_encoder=true fgc.metric_learning=true fgc.epochs=60 $AUG
# R1a — encoder leave-out cho P1 (mục mở rộng):
fgc --train         fgc.global_encoder=true fgc.metric_learning=true fgc.epochs=60 $AUG \
                    'fgc.exclude_classes=[<danh sách H từ B5>]'
```
**2026-11-13**: thay `fgc.epochs=60` bằng `fgc.epochs=5 '++fgc.save_epochs=[3,5]'`, lấy
`__global__.ep3.pt` cho cả hai (mục "Quyết định 2026-11-13 — chọn checkpoint").

Đặt tên rõ ràng khi copy về Drive: `__global_lo__.pt` (R1a) và `__global__.pt` (R1b).

**Kiểm**:
- 32 thư mục class trong `crops_fgc/train` (có cả Symbol_20, 21, 24, 25).
- Log `[fgc/__global__] 32 fine classes, ...` (R1b) / `32 − K fine classes` (R1a).
- `val_acc` cao (DigitizePID là synthetic, Stage 1 đạt 0.995 — closed-set 32 class nên > 0.95;
  thấp hơn nhiều là dấu hiệu lỗi data, không phải model yếu).
- Ra `fgc_checkpoints/__global__.pt`. **Copy về Drive ngay** (session Colab ~6h).

---

## B4 — Chạy GĐ2 + đọc separation 🧑🤖

```
prototypes oneshot.encoder_ckpt=<run-R1a>/__global_lo__.pt oneshot.prototypes=.../iso_prototypes_lo.pt
prototypes oneshot.encoder_ckpt=<run-R1b>/__global__.pt    oneshot.prototypes=.../iso_prototypes.pt
```

Với mỗi encoder, đọc dòng log `prototype separation: mean off-diagonal cos=…, max=… (closest pair: A vs B)`.

| Kết quả | Ý nghĩa | Làm tiếp |
|---|---|---|
| max < ~0.8 | Template ISO tách được trong space DigitizePID | Bỏ qua GĐ3 |
| max ≥ ~0.9 | Có cặp ISO gần như trùng | 🧑 xem 2 PNG đó: thật sự giống nhau (vd. valve_general vs valve_gate) hay encoder không transfer? |
| mean cao (≥ ~0.6) | Cả space co cụm — encoder không transfer sang ISO | So với bản prototype có stub (B9); vẫn kém → `docs/BACKUP_PLAN.md` (B7/GĐ3 đã bỏ) |

Các ngưỡng trên là **ước lượng ban đầu**, ghi lại số thật vào HANDOFF — chính các số này
là một bảng trong luận văn (separation với prototype sạch / có stub).

**Kết quả R2 đầu tiên (2026-11-12, R1b epoch 3, prototype sạch, 1 view)** — chi tiết HANDOFF §6d:
- mean 0.245, max 0.989 (pump_general vs vessel_spherical); 12/861 cặp > 0.9. Mean ĐẠT, max KHÔNG.
- **S–S gần nhất 0.389**: không cặp S–S nào sát. Mọi cặp > 0.85 đều dính lớp V hoặc H.
- Hai cụm: van "chấm + vạch" (globe/butterfly 0.981, check_valve_globe/swing 0.921) và hình tròn
  (bơm, máy nén, bình cầu, kính quan sát; toàn H tự nhiên).
- Đã xem PNG theo hàng "max ≥ 0.9": các cặp KHÔNG giống nhau với mắt người → encoder không chuyển
  được. Nguyên nhân cụm van đã xác nhận: đặc điểm không cộng dồn (gate/general 0.133 nhưng
  globe/butterfly 0.981 — DigitizePID không có lớp nào mang cả chấm lẫn vạch). Không phải tiền
  xử lý.
- Lề template: scale 1.2 (khớp lề crop) cho mean 0.240, max 0.990 → separation không nhạy với lề.

---

## B5 — Nhãn ISO + chọn danh sách giữ ra H 🧑 ⛔ chặn R1 (không chỉ B8)

**Vấn đề gốc**: GT của DigitizePID là `Symbol_N`, không phải ISO. Muốn đo "nhận đúng symbol
ISO" thì cần biết crop nào **là** symbol ISO nào. `docs/symbol_mapping/symbol_map.csv` đang
có nhiều dòng `TODO`.

**Việc**: 🧑 hoàn thành cột `fine_name` (tên ISO trong `vocab.csv`) cho 32 dòng. Mỗi dòng rơi
vào một trong hai loại:
- **Map được** sang 1 symbol ISO → tập **in-vocab**: đo one-shot accuracy.
- **Không có trong ISO** (plug valve edrawmax, instrument bubble ISA-5.1, …) → để trống /
  đánh dấu `OOV` → tập **out-of-vocab có sẵn**: đo "Other" recall.

Đây là cách lấy tập test "Other" mà **không cần thu thêm data** — đóng góp chính có số đo thật.

**Công cụ (2026-10-19)**: `docs/symbol_mapping/review_b5/` — mỗi class 8 crop thật + template
đang map (`classes_*.png`), toàn bộ template (`iso_templates_all.png`). Cột `status` dùng
`ISO` / `OOV` / `VARIANT` / `UNSURE` (thay TODO/DONE). VARIANT = cùng nghĩa với một
symbol ISO nhưng vẽ theo quy ước khác (vd. S2: ball valve có X trong vòng, ISO X8071 vòng trơn;
có `reg_number`); VARIANT và UNSURE bị loại khỏi H — VARIANT dùng cho phân tích phụ về
quy ước vẽ, tách khỏi P1. Kiểm bằng
`uv run python scripts/check_b5.py` (lỗi = chưa được train R1).

**Quyết định template (user chốt 2026-10-22, TRƯỚC khi có kết quả)**: S17–S19 (blind disc)
được DigitizePID vẽ gắn trên cặp mặt bích (= S21 / ISO 511), template ISO chỉ có phần chỉ thị
đĩa → dùng **template ISO trơn** (phương án A) cho P1. Template ghép (chỉ thị + 511) chỉ là
ablation tùy chọn, theo quy tắc cố định "symbol ISO đặt trên mối nối mặt bích thì ghép 511".

**Chọn danh sách H** (đã chốt: leave-classes-out là protocol CHÍNH — xem mục "Protocol đánh
giá"). Làm ngay sau khi map xong, TRƯỚC R1, TRƯỚC khi thấy bất kỳ kết quả nào:
- **H_in — 6–8 class map được sang ISO**, trải nhiều họ (không chỉ valve: có blind_disc / HX
  nếu map được), và **ít nhất 1–2 ca khó**: class bị giữ ra mà "hàng xóm" ISO gần nó vẫn còn
  trong train (vd. giữ ra globe valve nhưng train gate valve) — đo được khả năng phân biệt
  tinh, không chỉ nhận dạng thô.
- **H_out — 2–4 class OOV** (vd. instrument bubble ISA, plug valve edrawmax) → tập "Other"
  thật sự unseen.
- Đừng rút cạn một họ: mỗi coarse family vẫn còn ít nhất 1 class trong train.
- Ghi H vào `oneshot.holdout.h_in` / `h_out` trong `configs/config.yaml` (KHÔNG vào
  `fgc.exclude_classes` — mặc định đó áp cho mọi run, kể cả R1b đủ 32 class; notebook tự
  dựng override cho R1a từ `oneshot.holdout`) + HANDOFF kèm lý do chọn từng class. Push.
- (Tùy quota Colab) 2–3 bộ H khác nhau → báo cáo trung bình ± độ lệch, chắc hơn 1 bộ.

**Trạng thái (2026-11-09)**: `symbol_map.csv` xong 32/32 (10 ISO / 11 OOV / 9 VARIANT / 2
UNSURE); `oneshot.holdout.h_in` / `h_out` vẫn rỗng — đây là việc còn lại của B5.

**✅ H ĐÃ CHỐT 2026-11-11** (chi tiết + lý do: HANDOFF §6c; nguồn: trang kiến trúc
https://claude.ai/artifact/BtJxRzz8PdFjXVhciZiCo6):
- **Chia A (chính, trong `config.yaml`)**: `h_in` = S3, S10, S5, S4, S18, S19; `h_extra` = S2,
  S13, S12; `h_out` = S9, S15 (khó), S27, S30 (dễ). R1a còn 19/32 lớp. `check_b5.py` sạch.
- **Chia B (dự phòng, đăng ký trước, chỉ ghi trong HANDOFF)**: `h_in` = S1, S10, S5, S17, S18;
  `h_extra` = S2, S13; `h_out` = S7, S16, S26, S32. Chạy nếu còn quota → báo trung bình ± độ lệch.
- User muốn "chỉnh H sau khi có kết quả sơ bộ" → **không**: chỉnh sau khi thấy số là
  cherry-picking và phải train lại R1a. Thay vào đó đăng ký trước hai bộ.
- `h_extra`: notebook đã dựng R1a với `h_in + h_out + h_extra` và finalize kiểm cả `h_extra`
  (2026-11-11) → R1a không còn bị chặn; B5b chỉ còn phần kiểm luật trong `check_b5.py`.

**Luật mới — anh em VARIANT (2026-11-09)**: hai lớp ISO có lớp VARIANT cùng `reg_number`:
`valve_general` (S4 ↔ S12, 2101) và `valve_ball` (S5 ↔ S2, S13, X8071). Giữ S5 ra mà S2/S13 vẫn
train thì encoder vẫn học "van bi" qua biến thể → `valve_ball` không thật sự chưa thấy. Chọn S4
hoặc S5 vào `h_in` thì **phải giữ ra luôn các VARIANT cùng `reg_number`** (chúng không vào tập
test, chỉ bị bỏ khỏi train). Cần thêm một key, vd. `oneshot.holdout.h_extra`, để R1a bỏ chúng
mà P1 không test chúng.

**Ứng viên `h_in` sau mọi luật**: S1 gate, S3 globe, S10 needle; S4 general (+S12), S5 ball
(+S2, S13); tối đa 2 trong 3 lớp blind disc S17/S18/S19. **Không chọn được**: S21 flange và S24
flow_direction (họ một lớp; giữ ra là rút cạn họ — `check_b5.py` báo lỗi). Ứng viên `h_out`: 11
lớp OOV (S7, S9, S15, S16, S26–S32), không rút cạn họ `instrument`.

## B5b — `check_b5.py` kiểm luật VARIANT 🤖 (Phase A, nhỏ) — ✅ 2026-11-13

**Đã làm.** `utils.oneshot.holdout_classes` trả `Holdout(h_in, h_out, h_extra)` (`.tested` =
`h_in + h_out` cho P1, `.excluded` = cả ba cho R1a); một lớp nằm ở hai list là lỗi. `check_b5.py`
báo ERROR khi một lớp `h_in` có VARIANT cùng `reg_number` chưa nằm trong `h_extra`, khi một lớp
`h_extra` không phải VARIANT của một lớp `h_in`, và tính `h_extra` vào luật "rút cạn họ". Notebook
đọc `h_extra` qua `holdout_classes`. Chia A: 0 lỗi; bỏ S12 khỏi `h_extra` → lỗi đúng chỗ.
`tests/test_check_b5.py` có test chạy trên `symbol_map.csv` + `config.yaml` thật. Nếu vấn đề mở 5
đổi S21 sang VARIANT thì chạy lại `check_b5.py`.

Nội dung gốc:

- ERROR khi một lớp trong `h_in` có VARIANT cùng `reg_number` không nằm trong H (`h_extra`).
- `oneshot.holdout.h_extra` (mặc định `[]`): `holdout_classes` trả thêm danh sách này; notebook
  dựng `fgc.exclude_classes = h_in + h_out + h_extra`; P1 chỉ test `h_in` + `h_out`.
  ✅ Phần notebook xong 2026-11-11 (cell 4 đọc `h_extra` thẳng từ config, `HELD`; finalize kiểm
  `HELD`; `tests/test_colab_notebook.py`). Còn: luật trong `check_b5.py` + đưa `h_extra` vào
  `holdout_classes` để B6b dùng chung một nguồn.
- Test: S5 trong `h_in` thiếu S2/S13 → lỗi; có đủ → sạch; `h_extra` không vào tập test P1.

Ngoài ra: `data/raw/real/` đang trống. Nếu có P&ID **thật** (scan/bản vẽ dự án) thì đó là
tập test cross-domain mạnh hơn DigitizePID synthetic — ghi vào HANDOFF §3 khi có.

---

## B6a — GĐ4: one-shot inference 🤖 (Phase A) — ✅ 2026-11-13

**Đã làm** (`src/pipeline/oneshot_infer.py`, `tests/test_fgc_infer_oneshot.py`, 18 test):
- `pid-fgc-infer ++fgc.oneshot=true oneshot.encoder_ckpt=<enc> oneshot.prototypes=<dir>`: mỗi
  crop (`cut`, box + 10%) → `query_views` → embed → `match_views` với prototype ISO. Box dưới
  `fgc.route.min_score` hoặc crop rỗng → `__unrefined__`.
- Ra file MỚI `derived/detections_oneshot/<split>/<id>.txt`, 10 cột:
  `label best_name cx cy w h det_score best_cos second_cos coarse_id`. `label` = quyết định ở
  `oneshot.other_min_cosine` (chỉ để đọc nhanh); `best_name`, `best_cos`, `second_cos` **chưa cắt
  ngưỡng** → B6b quét τ / AUROC / OSCR / ablation margin top-1–top-2 không cần chạy lại suy luận.
  Đọc bằng `read_oneshot_detections`. Dòng `__unrefined__` có cosine `nan`: B6b phải bỏ, không
  tính là "bị loại".
- Nghĩa của `best_cos`: `aggregate=mean` → cosine của vector trung bình các view (quét τ trên nó
  tái tạo đúng `label`). `aggregate=vote` → trung bình cosine của các view thắng; quét τ trên nó
  chỉ **xấp xỉ** phiếu bầu có cắt ngưỡng (đúng khi các view đồng ý; P2 thu nhỏ: vote ≈ mean).
  → Đề xuất B6b chốt `aggregate=mean` cho đường quét τ, hoặc ghi rõ xấp xỉ.
- **Chặn encoder ≠ prototype**: `pid-iso-prototypes` giờ ghi thêm `encoder_sha256`; lúc suy luận so
  lớp train, lớp bị loại, backbone, embed_dim, imgsz và sha256 với encoder đang nạp → lệch là lỗi.
  File prototype cũ (chưa có sha256) vẫn nạp được, vẫn so các trường còn lại. Lưu ý: resume /
  lưu lại checkpoint đổi sha256 → phải dựng lại prototype.
- `oneshot.exclude_prototypes` (mặc định `[]`): bỏ template khỏi thư mục trước khi so (cho vấn đề
  mở 4 — chưa quyết bỏ gì). `oneshot.route_by_coarse` (mặc định tắt) + `oneshot.coarse_map`
  (bắt buộc khi bật); lớp Stage 1 không có trong map, hoặc map tới toàn template đã bị bỏ → so cả
  thư mục, log một lần.
- Khoá mới đọc bằng `.get` (dùng `++` được); **chưa ghi vào `config.yaml`**, chờ duyệt:
  `fgc.oneshot: false`, `oneshot.exclude_prototypes: []`, `oneshot.route_by_coarse: false`,
  `oneshot.coarse_map: null`.

Nội dung gốc:

**Quyết định vocab (đề xuất)**: ở chế độ one-shot, **prototype directory chính là vocab của
Stage 2**. Không sinh `classes.yaml` mới — `classes.yaml` vẫn là vocab của Stage 1 (8 coarse)
và của GT DigitizePID.

**Code**:
- `configs/config.yaml`: `fgc.oneshot: false`.
- `fgc_infer.py`:
  - Khi `fgc.oneshot=true`: load encoder `__global__.pt` qua `load_arcface_encoder` +
    `PrototypeDirectory.load(oneshot.prototypes)`; **bỏ** guard `global_encoder` cho nhánh này.
  - `_classify_crop` nhánh one-shot: `make_views(crop, query_views)` → `embed_images` →
    `match_views(min_cosine=oneshot.other_min_cosine, strategy=oneshot.aggregate)`.
  - **Routing theo coarse**: coarse của Stage 1 là DigitizePID (8), coarse của prototype là ISO
    (12) — phải có bảng map. Thêm `oneshot.coarse_map` (vd. `valve → [valve, check_valve]`,
    `instrument → [agitator]`, `reducer/flange/flow_direction → [piping]`) rồi
    `subset_by_coarse(...)`. Nguồn: bảng coarse map trong HANDOFF §3.
    Coarse không có trong map → match toàn bộ directory (ghi log).
  - **Routing là tuỳ chọn, mặc định TẮT** (2026-11-11, V8_REVIEW §2.1): cờ kiểu
    `oneshot.route_by_coarse: false` → so với cả directory. Bắt buộc tắt cho P4 / mọi bản vẽ
    ISO: detector DigitizePID gán bơm, tháp, agitator vào lớp thô gần nhất nó biết (vd.
    `valve`), routing khi đó loại hẳn prototype đúng. Chỉ bật cho DigitizePID như một ablation.
  - Output: ghi **thêm** file `derived/detections_oneshot/<split>/<id>.txt` với cột tên ISO
    (string) thay vì sửa format 7 cột hiện có — giữ nguyên contract cho eval cũ.
- `tests/test_fgc_infer_oneshot.py`: dùng `_FakeEncoder` + directory giả như `test_oneshot.py`;
  kiểm crop OOV → `__other__`, coarse_map giới hạn đúng tập ứng viên, file output đúng format.

---

## B6b — GĐ5: eval one-shot + "Other" 🤖 (Phase A)

Eval ở **mức crop** trước (như `compute_fgc_metrics` hiện tại và như protocol OSSR-PID) — tách
được lỗi Stage 2 khỏi lỗi detection. Chỉ số P1 mới đặt cạnh 85.78% của OSSR-PID được.

**Code** — `src/pipeline/evaluation.py`, method mới `compute_oneshot_metrics()`:
- Cấu hình `eval.oneshot.protocol: p1 | p2 | p3` — mỗi protocol một bảng riêng trong
  `metrics.json`, không gộp.
- Input chung dạng `<root>/<class>/*.png` + CSV map class → ISO fine_name / OOV, để P3 (ASU)
  cắm vào sau mà không phải sửa code. P1/P2: `crops_fgc/<split>/Symbol_N/` + `symbol_map.csv`.
- **P1**: chỉ lấy crop của class H (`h_in`, `h_out`; không `h_extra`), từ cả hai split — τ
  không nhìn chúng (2026-11-11), nên không cần tách phần calibrate. Ghi số instance. **Guard rò
  rỉ**: nếu class test nào nằm trong `fine_classes` của checkpoint encoder → raise, không chạy.
- **In-vocab**: top-1 / top-3 one-shot accuracy, confusion theo ISO fine_name, per-coarse.
- **"Other"** (in-vocab = negative, OOV = positive): **AUROC** trên max-cosine và **OSCR**
  (độ chính xác lớp biết theo tỉ lệ nhận nhầm lớp lạ, quét mọi τ) — hai số so sánh chính,
  không phụ thuộc ngưỡng; precision / recall / F1 tại τ đã chọn (B8).
- **Đường quét τ** 0.0→1.0: accuracy lớp biết (sau reject) vs tỉ lệ loại lớp lạ → PNG. Bắt
  buộc với P4. Đây là hình báo cáo, **không** phải chỗ chọn τ.
- **Chọn τ** (`oneshot.other_min_cosine`, thay placeholder 0.35): hàm riêng, chỉ đọc
  `val_sel` DigitizePID, lớp đã thấy — quy tắc A và B ở B8. Test: hàm chọn τ từ chối khi nhận crop
  của lớp H hay ảnh OPEN100-ISO.
- **Box sai của detector**: box dự đoán có IoU < 0.3 với mọi GT trên `val_sel` DigitizePID →
  tỉ lệ bị τ loại (cần detector chạy trên val).
- Ablation sẵn có qua config, không cần code thêm: `aggregate=vote|mean`,
  `query_views` 1 view vs 8 view, `fgc.aug.subpart.p` 0 vs 0.5 (cần 2 lượt train).
- Ghi vào `metrics.json` dưới key `oneshot`.
- `tests/test_oneshot_metrics.py`: AUROC = 1.0 khi tách hoàn hảo, 0.5 khi ngẫu nhiên; sweep
  đơn điệu; tập OOV rỗng → bỏ qua metric "Other" có cảnh báo, không crash; **P1 với class
  test có trong checkpoint → raise**; OSCR = 1 khi tách và gọi tên hoàn hảo; τ@TPR95 chấp
  nhận đúng 95% crop biết; quy tắc B không bao giờ dùng ★ của chính lớp đó; hàm chọn τ từ
  chối crop lớp H / OPEN100-ISO.

**Thêm 2026-11-09 — P4 (OPEN100-ISO)**:
- Input: export V8 (`data/iso_open100/canonical/`, COCO — data root riêng, đọc bằng
  `data_root=data/iso_open100`), crop theo box GT với cùng lề như
  `extract_crops` (`CROP_MARGIN_FRAC`), tách `images=clean` và `images=degraded`.
- Mỗi instance gắn nhóm **S / V / H chọn / H tự nhiên / Other** bằng một hàm duy nhất, tính từ
  `symbol_map.csv` + `fine_classes` / `excluded_classes` của checkpoint (bảng "Lớp ISO thuộc
  nhóm nào"). Không có danh sách gõ tay.
- Báo cáo mỗi nhóm một dòng (top-1, top-3), không gộp; "Other" với dương = `other_iso`, âm = mọi
  lớp vocab; `other_isa` và `other_generic` báo riêng.
- Calibrate ngưỡng **không** dùng OPEN100-ISO (chỉ dùng ngưỡng đã chốt từ P1) — OPEN100-ISO
  là đề thi.
- **Quy tắc báo cáo (2026-11-11, V8_REVIEW §2.2)** — áp cho mọi số trên OPEN100-ISO (P4, Stage 1
  không phân lớp, end-to-end):
  - **Không bao giờ gộp ảnh sạch và suy giảm thành một số.** Export giữ `images: both`; khi chấm
    tách theo id (ảnh suy giảm có hậu tố `_deg`). **Số chính = ảnh suy giảm**; ảnh sạch là
    tham chiếu để đo mức giảm do nhiễu.
  - **Báo cáo theo tờ gốc** (`iso00`…`iso11`): mỗi tờ một số (gộp 36 biến thể của tờ đó), rồi
    trung bình ± độ lệch trên 12 tờ. 36 biến thể của một tờ không phải mẫu độc lập; mẫu độc
    lập thật chỉ là 12 topology.
  - Kèm số instance mỗi nhóm để người đọc biết số nào dựa trên ít mẫu.
- End-to-end trên cả tờ (Stage 1 + Stage 2): TP khi IoU ≥ 0.5 **và** đúng nhãn ISO.

**Thêm 2026-11-11 — Stage 1 không phân lớp (V8_REVIEW §2.1)**, trong `compute_detection_metrics`:
- Tuỳ chọn kiểu `eval.class_agnostic: false` (mặc định giữ hành vi cũ). Bật → mọi box GT và mọi
  box dự đoán thành một lớp "symbol"; P, R, mAP@50, mAP@50:95 như cũ. Tách `images=clean` /
  `degraded`.
- **Recall theo nhóm**: mỗi box GT mang nhóm S / V / H tự nhiên / Other (cùng hàm chia nhóm của
  P4, đọc lớp mịn ISO trong COCO) → recall từng nhóm và từng lớp mịn. Nhóm H tự nhiên trả lời
  "detector DigitizePID có tìm được hình chưa từng thấy không" — trần của end-to-end.
- Bảng ánh xạ 8 lớp thô (V8_REVIEW §2.1) chỉ làm nếu cần báo cáo phụ; không chặn gì.
- Tests: bật/tắt cho cùng số liệu khi chỉ có một lớp; dự đoán đúng box nhưng sai lớp được tính
  TP khi bật, FP khi tắt; recall theo nhóm cộng lại khớp recall tổng.
- Nếu recall H tự nhiên thấp: ghi thành kết quả, và là lý do để làm V9 (train Stage 1 trên bản
  vẽ ISO sinh từ 750 sơ đồ tổng hợp, OPEN100 vẫn là test) — quyết định của user.

---

## B6c — Closed-set với `__global__.pt` 🤖 (Phase A) — lõi luận văn (2026-11-09) — ✅ 2026-11-13

**Đã làm** (`tests/test_closed_set_global.py`):
- `fgc_infer` với `fgc.global_encoder=true`: nạp `<fgc.ckpt_dir>/__global__.pt`, cột lấy từ
  `fine_classes` của checkpoint. Mọi box có score ≥ `fgc.route.min_score` đều được phân lớp,
  kể cả họ 1:1 (flange, flow_direction…), không nhìn lớp thô. `++fgc.route.by_coarse=true` giới
  hạn softmax trong họ của Stage 1 (ablation "routing theo coarse có giúp không"). **Không có
  cổng Other** ở nhánh này: Other của encoder toàn cục là việc của B6a, ngưỡng theo B8.
- `evaluate eval.mode=fgc_only` (mới): chấm head trên crop GT, không cần detector. Ra top-1/top-3,
  macro-F1, rare macro-F1, P/R/F1 từng lớp, confusion đầy đủ (JSON + PNG), theo họ coarse
  (accuracy, `cross_family` = tỉ lệ đoán sang họ khác), `other_gate` (cosine max, tham khảo).
- `det_only` chấm GT theo `detector.label_level`: YOLO 32 lớp fine giờ chấm được theo lớp fine.
  `detect_infer` ghi id fine cho detector fine và kiểm tên lớp trong checkpoint khớp
  `classes.yaml` (trước đó map id fine qua 8 tên coarse: sai tên hoặc IndexError).
- mAP bỏ dự đoán không có nhãn vocab (`-1` chưa tinh chỉnh, `-2` Other). Trước đó mỗi sentinel
  thành một "lớp" AP 0, kéo mAP xuống. mAP50:95 đã có sẵn (`eval.iou_thrs`).
- `fgc_infer` từ chối file detection của detector fine (`detector.label_level=fine`).
- Notebook mục 8: chạy C ngay sau R1b.

**Chưa làm**:
- Chấm trên `test` thay vì cả `val`: `fgc_only` hiện đọc `crops_fgc/val` (100 ảnh). Số C chính
  thức phải trên 50 ảnh `test` (mục "Tách val") → cần lọc crop theo id ảnh từ
  `digitizepid_val_test.json` (tên crop `<id ảnh>_<n>.png`), đi cùng việc commit hai file split.
  Với C đã bão hoà, số trên cả `val` dùng tạm để chẩn đoán được.
- C trên crop DigitizePID suy giảm (`scan_degrade`, seed cố định) — "nếu kịp".

Nội dung gốc:

Hiện `fgc_infer` từ chối global encoder và `evaluation.compute_fgc_metrics` lặp theo
`fgc_groups`, nên phần lõi (protocol C) chưa chạy được với encoder dùng chung.
- `fgc_infer`: nhánh closed-set cho `__global__.pt` — softmax trên head 32 lớp, ghi
  `detections_fgc/` đúng format 7 cột hiện có.
- `evaluation`: `compute_fgc_metrics` cho checkpoint global — top-1, top-3, macro-F1, confusion
  32 lớp, theo họ coarse.
- End-to-end mAP@50 / mAP@50:95 theo 32 lớp fine (`eval.mode=det_plus_fgc`) để đặt cạnh YOLO
  một-stage train với `detector.label_level=fine` (code đã có, chỉ cần một lượt train Colab).
- Phương án rẻ hơn nếu không kịp: dùng head per-group (đã chạy được) cho riêng chương lõi. Khi
  đó encoder closed-set và encoder one-shot là hai model khác nhau — phải ghi rõ.

---

## B6d — Chuẩn bị đề thi OPEN100-ISO 🤖 (Phase A, ⛔ trước R4) — 2026-11-11

Hai việc từ V8_REVIEW, quan trọng hơn sau khi OPEN100-ISO thành đề thi chính:

**1. Kiểm ảnh suy giảm với config V7 hiện tại (V8_REVIEW §2.5).** Export kiểm lại ảnh sạch (vẽ
lại rồi so box) nhưng dùng nguyên JSON và ảnh `degraded/`. V6/V7 đã được sinh lại nhiều lần
(2026-11-07/08), nên nếu V7 chạy với config khác mà quên chạy lại, đề thi sẽ lệch mà không ai
biết.
- Trong export: với mỗi ảnh suy giảm, sinh lại `sample_params` từ seed lưu trong JSON và config
  `render_iso.degrade` hiện tại, so với `params` đã lưu. Lệch → lỗi, ghi tên ảnh, không export.
- Test: config đổi một dải → export báo lỗi; config giữ nguyên → qua.
- Thay cho quy tắc tay "sinh lại V6/V7 thì phải export lại" (§3 câu 4): quên thì export tự bắt.

**2. Vùng bỏ qua cho mũi tên dẫn và cờ (V8_REVIEW §2.6).** Export đang bỏ **12,096** box
(`leader_arrow` và cờ inlet/outlet không có nhãn), so với 93,800 box được giữ. Khi chấm không phân
lớp, detector DigitizePID (có lớp `flow_direction`) rất dễ bắt mũi tên dẫn, và mỗi lần bắt là một
dương giả.
- Export: ghi các box này vào COCO với `iscrowd: 1` (cách chuẩn của COCO cho vùng bỏ qua), lớp
  riêng không có trong vocab. `drop_labels` đổi nghĩa thành "ghi là vùng bỏ qua".
- `evaluation`: dự đoán khớp vùng bỏ qua (IoU ≥ 0.5, hoặc nằm phần lớn trong vùng) không tính FP,
  cũng không tính TP; GT bỏ qua không vào mẫu số recall. Kiểm `build_manifest` / `coco_to_yolo` /
  `extract_crops` bỏ qua `iscrowd: 1` (không thành nhãn train, không thành crop).
- Báo cáo thêm: số dự đoán rơi vào vùng bỏ qua (cho biết detector bắt mũi tên dẫn nhiều cỡ nào).
- Test: dự đoán trùng vùng bỏ qua → precision không đổi; GT bỏ qua không làm giảm recall; DigitizePID
  (không có `iscrowd`) cho kết quả y như trước.

---

## B6e — Stage 1 không phân lớp lúc suy luận 🤖 (Phase A, ⛔ trước R4) — 2026-11-11

**Vì sao**: WBF trong `detect_infer` (`ensemble_boxes.weighted_boxes_fusion`) chỉ gộp box
**cùng nhãn**, NMS của YOLO mặc định cũng theo lớp. Trên bản vẽ ISO, detector coarse có thể ra
hai box chồng nhau khác nhãn cho một ký hiệu lạ (vd. `valve` + `heat_exchanger`): chấm không
phân lớp thì một box thành FP, end-to-end thì thành hai crop.

**Code**:
- `detector.agnostic_merge` (mặc định `false` → số DigitizePID không đổi; bật bằng `++` override,
  không sửa `config.yaml`): YOLO `agnostic_nms=True` mỗi tile; WBF với mọi nhãn coi là một, mỗi
  box gộp giữ nhãn của box thành phần điểm cao nhất.
- **Tuỳ chọn** `data.label_level=agnostic` ở `coco_to_yolo`: mọi nhãn YOLO thành lớp 0 → train
  detector một lớp (một lượt Colab, sau R1b / R1a / YOLO 32 lớp nếu còn quota).
- Tests: hai box chồng khác nhãn → một box khi bật, hai khi tắt; nhãn giữ theo điểm cao nhất;
  `label_level=agnostic` ghi toàn lớp 0 và `data.yaml` có `nc: 1`.

**Bảng ablation Stage 1 (luận văn, phần "detection")**:

| Detector | Trạng thái | Đo |
|---|---|---|
| 1 lớp (agnostic) | tuỳ chọn, cần `label_level=agnostic` + 1 lượt train | recall nhóm H tự nhiên (P4), mAP DigitizePID không phân lớp |
| 8 lớp coarse | ✅ đã train | như trên + mAP coarse DigitizePID |
| 32 lớp fine | trong plan (baseline C) | như trên + mAP fine (so với hai-stage) |

Chọn detector mặc định sau khi có số: recall nhóm H tự nhiên cao hơn mà mAP DigitizePID không
giảm đáng kể. Không đủ quota → bỏ detector một lớp; coarse + `agnostic_merge` vẫn đủ.

**Ghi chú — có thể làm sau (user, 2026-11-11)**: trang kiến trúc
(https://claude.ai/artifact/BtJxRzz8PdFjXVhciZiCo6, mục Stage 1) đề xuất **detector một lớp làm
mặc định**, coarse làm ablation. Tạm thời giữ như plan (coarse đã train + `agnostic_merge`), nhưng
nếu làm detector một lớp thì theo trang:
- Dùng `single_cls=True` của Ultralytics thay cho code `label_level=agnostic` (không phải ghi lại
  nhãn) → phần `label_level=agnostic` ở trên có thể bỏ.
- **Vùng bỏ qua cho ký hiệu chưa gán nhãn trong DigitizePID** khi train detector, để detector
  không học coi chúng là nền. Trước hết phải kiểm DigitizePID có ký hiệu chưa gán nhãn không.
- **Lấy mẫu cân bằng theo nhãn gốc** (`flow_direction`, flange rất đông) dù detector chỉ có một
  lớp.
- Lý do trang đưa ra: Stage 1 chỉ cần trả lời "có ký hiệu không"; tên do Stage 2 gán.
- Kích hoạt lại khi: recall nhóm H tự nhiên của detector coarse thấp ở R4, hoặc khi còn quota.

---

## B7 — GĐ3: fine-tune với template ISO — ❌ BỎ (2026-11-09)

Train encoder trên template ISO trái với quyết định "template ISO chỉ dùng lúc suy luận", làm
mất "thêm ký hiệu không cần train lại" (quay về giới hạn của OSSR-PID), và nếu dùng template
của H thì H không còn là chưa thấy. Thay bằng B9. Thiết kế cũ còn trong git history (trước
2026-11-09).

---

## B9 — Prototype có stub, cách (d) 🤖 — ⏸ HOÃN sang Phase C (sau R4), quyết định 2026-11-10

Thiết kế giữ nguyên bên dưới. Làm khi đã có số R4 với prototype sạch; nên ưu tiên nếu P4 cho
thấy crop OPEN100-ISO có ống nối nhận kém hơn hẳn template sạch, hoặc R2 separation kém.

**Vì sao**: template ISO là hình sạch; crop từ detector dính đoạn ống ở cổng nối, mờ, lệch tâm.
Thay vì dạy encoder (đã bỏ, B7), làm phía prototype giống crop hơn. Không train gì.

**Cách làm**: mỗi template sinh `1 + K` view: bản sạch + K bản "bẩn"; prototype = trung bình
các vector (L2-normalize sau khi cộng). Mỗi bản bẩn:
1. **Ống ở cổng nối thật**: cạnh cổng (`L/R/T/B`) và vị trí (`u, v`) lấy từ template render
   (`utils.iso_templates.load_templates`, khớp theo `reg_number`; ví dụ van `L,R`, tháp
   `B,L,R,T`, agitator `T`). Độ dày ống = độ dày nét đo trên PNG
   (`scan_degrade.stroke_width`). Độ dài ngẫu nhiên, đến mép khung crop.
2. **Mờ / nhiễu nhẹ**: `ScanDegrade(p=1)` với dải hẹp (dùng lại, không viết mới).
3. **Lệch tâm**: dịch nhẹ trong khung, như box detector lệch.

**Code**:
- `configs/config.yaml` `oneshot.template_views.stubs: {n_views: 0, ...}` — `0` = hành vi hiện
  tại y nguyên (baseline).
- `utils/oneshot.py` (phía support của `make_views`) + `pipeline/iso_prototypes.py`; ghi
  `n_views` và seed vào `meta` của file prototype.
- **Template không có cổng**: 11 lớp vocab không có template render (`tank`, `motor_electric`,
  `hx_*`, `open_disc`, `pipeline`, `hose`, …) → chỉ dùng view sạch, ghi log. Riêng `open_disc`
  (S17, ứng viên `h_in`): 🧑 quyết định trước R2 — bổ sung cạnh cổng bằng tay (bảng nhỏ) hoặc
  để nguyên view sạch.
- **Kiểm khớp toạ độ**: PNG prototype (`iso_symbols_clean`) và SVG render cùng nguồn BK-2020,
  nhưng khung có thể khác → map `u, v` qua bbox mực của PNG; 🧑 duyệt một contact sheet (mỗi
  template: sạch + 3 bản bẩn) trước khi dùng, như B1.
- Tests: `n_views=0` cho prototype trùng bản hiện tại; ống chạm đúng cạnh cổng; cùng seed cùng
  kết quả; template không cổng → chỉ view sạch + log.

**Ablation (R2, R4)**: prototype sạch vs có stub, đo trên **cả** P1 (DigitizePID) và P4
(OPEN100-ISO). Nếu chỉ P4 tăng: có thể chỉ khớp với cách bộ render vẽ ống — phải ghi rõ.

**Gộp vào Phase C — `open_disc` 0 mẫu trên OPEN100-ISO (V8_REVIEW §2.4, 2026-11-11).** C0098
`open_disc` (lớp S, S17, ứng viên `h_in`) nằm ở `scope: extra` trong taxonomy nên không được
vẽ; ghi chú taxonomy: "no stubs: the pipe passes through". Cùng gốc với việc thiếu cổng nối ở
B9, nên xử lý một lần: định nghĩa cổng cho C0098 (🧑 duyệt), đưa vào `html`, sinh lại V6 → V7 →
V8 (export sẽ tự kiểm nhờ B6d). Đến lúc đó, nếu S17 nằm trong `h_in` thì P1 vẫn đo được trên
DigitizePID, chỉ P4 thiếu lớp này.

---

## B4a — Baseline encoder chỉ-ImageNet 🤖 (tùy chọn, đề xuất)

- `scripts/make_imagenet_encoder.py`: dựng `ArcFaceModel` với `fgc.pretrained=true`, KHÔNG
  train, lưu theo đúng format checkpoint của pid-fgc-train (`model`, `fine_classes`, ...).
  Với resnet34 + `embed_dim=512` thì `proj` là Identity → embedding = feature ImageNet thuần.
- Kiểm: `load_arcface_encoder` đọc được; `pid-iso-prototypes` chạy được với ckpt này.
- Chạy đầu Phase B (CPU đủ, 39 ảnh): số separation "trước khi học P&ID" để so với R2.
- Ở R4, chạy cả eval P1 với encoder này: nó chưa thấy class P&ID nào nên là baseline
  unseen tự nhiên — chênh lệch P1(ImageNet) → P1(leave-out) là phần đóng góp của train DigitizePID.

---

## B8 — Chạy eval thật + chốt số 🧑🤖

0. **C** (encoder đủ 32 class, closed-set, B6c): top-1 / macro-F1 / confusion trên crop val;
   mAP end-to-end 32 lớp fine cạnh YOLO một-stage fine.
1. **P1** (encoder leave-out): AUROC + OSCR trên crop `h_in` / `h_out`; số tại τ của quy tắc A
   và B (chọn trước, xem "Ngưỡng Other" dưới đây, không tune lại).
2. **P2** (encoder đủ 32 class): τ riêng của encoder này, cùng quy tắc; báo cáo riêng, gắn nhãn
   "seen-class".
3. **P3** nếu đã có ASU: như P2, báo cáo riêng.
4. **P4** (OPEN100-ISO, cả hai encoder): τ của encoder tương ứng, theo nhóm S / V / H chọn / H
   tự nhiên, sạch vs suy giảm, cộng end-to-end; **luôn kèm đường quét τ**.

**Ngưỡng "Other" (τ) — thay 2026-11-11, theo trang kiến trúc
(https://claude.ai/artifact/BtJxRzz8PdFjXVhciZiCo6, mục "Ngưỡng Other")**. Nếu τ được chọn trên
dữ liệu test thì mọi số P1 mất giá trị. Ba lớp:

1. **So sánh bằng chỉ số không phụ thuộc ngưỡng**: AUROC của max-cosine (`h_in` vs `h_out`),
   OSCR, đường accuracy-biết / tỉ lệ loại-lạ. Mọi so sánh giữa phiên bản (R1a, R1b, prototype
   sạch / có stub) dựa trên AUROC và OSCR. AUROC cao mà OSCR thấp = tách biết / lạ tốt nhưng gọi
   tên sai trong nhóm biết.
2. **Chọn τ trên `val_sel` DigitizePID (50 ảnh, theo ảnh; quyết định 2026-11-12), theo quy tắc cố định**.
   Chỉ nhìn crop của các lớp **đã thấy có prototype** (status ISO, còn trong train của encoder
   đó). Không nhìn `h_in`, `h_out`, OPEN100-ISO. Báo cả hai quy tắc:
   - **Quy tắc A — τ@TPR95**: τ sao cho 95% crop val của lớp biết được chấp nhận. Chuẩn open-set,
     không cần lớp lạ; không biết trước loại được bao nhiêu lớp lạ.
   - **Quy tắc B — bỏ prototype**: với crop val lớp c, bỏ ★ của chính c, tính max-cosine tới các
     ★ còn lại → crop đó thành "lạ" mà không train thêm; τ tại điểm Youden (hoặc EER) giữa
     "biết" và "lạ giả lập". Lạ giả lập dễ hơn lạ thật → τ hơi lạc quan; ghi rõ.
     **Độ nhạy (2026-11-12)**: khi hai ★ gần trùng (vd. valve_globe/valve_butterfly 0.98), bỏ ★
     của S3 thì crop S3 khớp ★ butterfly gần như trọn → "lạ giả lập" trông như "biết", τ Youden
     bị đẩy lên. Báo thêm τ quy tắc B khi loại các cặp ★ có cos > 0.95.
   - Mỗi encoder chọn τ riêng, cùng quy tắc. Ghi τ vào `config.yaml` kèm comment (run, quy tắc).
   - **Hạn chế đã biết**: với R1a (chia A), lớp đã thấy có prototype chỉ còn **S1, S17, S21,
     S24** (4 lớp) → τ của R1a ước lượng trên ít lớp; ghi số crop dùng để chọn τ.
3. **Lớp lạ thứ hai: box sai của detector** (box dự đoán IoU < 0.3 với mọi GT trên val
   DigitizePID: ống, chữ, góc bảng) → báo tỉ lệ bị τ loại. Lọt nhiều thì thử dạng điểm 2.

**Dạng điểm số**: (1) một τ chung trên max-cosine — mặc định, làm trước; (2) thêm lề top-1/top-2:
loại nếu cos₁ < τ hoặc cos₁ − cos₂ < δ (δ chọn cùng quy tắc trên val); (3) ngưỡng theo từng
prototype — chỉ cho lớp S có val, lớp H vẫn τ chung; chỉ là ablation.

**Chuyển ngưỡng sang OPEN100-ISO**: τ chọn trên crop DigitizePID chưa chắc đúng cho OPEN100-ISO
(phong cách vẽ khác) → luôn kèm đường quét τ trên P4, ghi là hạn chế đã biết. Không tách một
phần OPEN100-ISO để hiệu chỉnh τ, trừ khi tách theo tờ gốc, công bố danh sách, và báo cáo như
một thiết lập riêng.

**Thứ tự**: (a) AUROC + OSCR trên P1 cho mọi phiên bản — bảng so sánh chính; (b) τ quy tắc A và
B trên val cho từng encoder; (c) P1, P2, P4 tại hai τ đó + đường quét; (d) kiểm τ trên box sai
của detector; nếu lọt nhiều, thêm lề top-1/top-2.

**Siêu tham số khác** (vd. của B9 sau này: số bản có stub, độ dài ống): chọn trên cùng split
`val` đó; không bao giờ trên P1 hay OPEN100-ISO.
5. Baseline ImageNet (B4a) chạy P1 và P4.
6. Bảng luận văn (phương án B):
   - Chương lõi: bảng C (hai-stage vs YOLO một-stage).
   - Chuyển sang ISO: bảng P4 theo nhóm và theo mức suy giảm; P2.
   - Mở rộng: P1 one-shot acc + "Other" AUROC/P/R, cạnh OSSR-PID 85.78% kèm câu nêu khác biệt
     protocol.
   - Ablations: vote/mean, 1 vs 8 view, subpart on/off; prototype sạch vs có stub (B9, nếu
     Phase C đã làm).
7. Cập nhật HANDOFF §0 + §6 với số thật.

**Ghi chú khi viết phần dữ liệu (V8_REVIEW, 2026-11-11)** — không sửa code:
- §2.3: trong data root OPEN100-ISO, `source=synthetic` và file `digitizepid_splits.json` là tên
  bắt buộc của `build_manifest` có sẵn; ảnh là bản vẽ lại (đúng nghĩa "synthetic"), không phải
  DigitizePID.
- §2.7: trục agitator không nhô lên khi có tag line ngay trên (hành vi V5b) → bộ test có cả hai
  dạng agitator. Nếu recall agitator thấp bất thường, xem lại điểm này khi phân tích.
- §3 câu 5: `by_source` không có tập `val` — chỉ cần thêm nếu làm V9.

---

## Checklist nhanh

**Phase A — Code**
- [x] B0 pytest + ruff xanh (207 pass, 2026-10-18)
- [x] B1 subpart augment `SubpartWarp` + user duyệt ảnh (2026-10-18)
- [x] B1b `fgc.exclude_classes` + tests chống rò rỉ (2026-10-18)
- [x] B2 run_pipeline `prototypes` + notebook global + zip mới + assert 32 class + cell leave-out (2026-10-19)
- [x] B6a GĐ4 one-shot infer + coarse_map (routing mặc định tắt; tắt bắt buộc cho P4) (2026-11-13)
- [ ] B6b GĐ5 metrics + sweep + AUROC, theo protocol P1/P2/P3/P4 + nhóm S/V/H tự động + guard rò rỉ
      + Stage 1 không phân lớp (`eval.class_agnostic`) + recall theo nhóm
- [x] B6c closed-set infer + eval cho `__global__.pt` (lõi, protocol C) (2026-11-13; còn lọc theo `test`)
- [ ] B6d đề thi OPEN100-ISO: kiểm ảnh suy giảm với config V7 + vùng bỏ qua (`iscrowd`) cho mũi tên dẫn / cờ — ⛔ trước R4
- [ ] B6e Stage 1 `agnostic_merge` (+ tuỳ chọn `label_level=agnostic`) — ⛔ trước R4
- [x] B1c crop có ngữ cảnh + box jitter + sửa lề crop lúc suy luận + notebook `AUG` (2026-11-11)
- [x] ~~B7-code GĐ3 fine-tune ISO~~ — bỏ 2026-11-09
- [ ] B4a-code baseline ImageNet (tùy chọn)
- [x] B5 🧑 hoàn thành `symbol_map.csv` (32/32, 2026-11-09)
- [x] B5 🧑 **chọn H_in, H_out** (theo luật VARIANT) — chia A, 2026-11-11
- [x] B5b 🤖 `check_b5.py` luật VARIANT + `oneshot.holdout.h_extra` (2026-11-13)

**Phase B — Chạy** (🧑 Colab, đúng thứ tự)
- [ ] (B4a) baseline ImageNet → separation "trước khi học"
- [ ] R1b = train `__global__.pt` (đủ 32 class, `AUG`), copy về Drive — chạy được khi H đã chốt
- [ ] YOLO một-stage 32 lớp fine (`detector.label_level=fine`) — baseline cho C
- [ ] (tuỳ chọn) YOLO một lớp (`label_level=agnostic`, B6e) — ablation Stage 1
- [ ] R1a = train `__global_lo__.pt` (leave-out H + `h_extra`), copy về Drive
- [ ] R2 = prototypes (sạch) cho cả hai encoder, ghi số separation
- [ ] R4 = B8: C (lõi) + P1 + P2 + P4 (+ P3) + baseline; AUROC/OSCR + τ quy tắc A/B trên val DigitizePID (lớp đã thấy)

**Phase C — Sau R4, tùy chọn** (hoãn 2026-11-10)
- [ ] B9 prototype có stub, cách (d) (`oneshot.template_views.stubs`, mặc định tắt) + 🧑 duyệt contact sheet + quyết `open_disc`
- [ ] R2' prototype có stub (encoder có sẵn, không train lại) → R4' P1 + P4 cùng ngưỡng
- [ ] (tuỳ chọn) cổng cho C0098 `open_disc`, đưa vào `html`, sinh lại V6 → V8 (V8_REVIEW §2.4)

## Lưu ý vận hành

- Máy local (2026-10-18): đã có `uv` + `.venv` (Python 3.11, torch CPU/MPS) → test/code chạy
  local bằng `uv run ...`; train vẫn trên Colab.
- Code session 2026-10-10 đang là **thay đổi chưa commit** trên branch
  `feat/prepare-for-colab-stage-2` (kèm fix ruff của B0) — commit trước khi bắt đầu B1.
- Repo đã là git. Không `git init` / xoá `.git`.
- Template ISO (`data/raw/iso_symbols_clean/`) sinh từ SVG của Steven Baltakatei Sandoval,
  BK-2020-04-PID-1, **CC BY-SA 4.0** — luận văn phải ghi nguồn. Tạo lại:
  `scripts/extract_iso_symbols_svg.py` (xem `docs/experiments/iso_templates_bk2020/`).
- Cả hai ngưỡng (`fgc.route.other_min_cosine`, `oneshot.other_min_cosine`) đều là
  placeholder 0.35 — **không** dùng chung số, chỉ `oneshot.*` được calibrate ở B8.