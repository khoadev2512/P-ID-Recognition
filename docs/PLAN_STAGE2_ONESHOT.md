# PLAN — Stage 2 one-shot: từ code GĐ2 đến kết quả luận văn (lập 2026-10-14)

Plan để **thực thi ở session sau**. Đọc kèm `docs/HANDOFF.md` (bối cảnh, quyết định đã chốt,
những gì đã thất bại). File này chỉ nói **làm gì, theo thứ tự nào, xong thì kiểm bằng gì**.

Ký hiệu: 🧑 = việc của user (Colab / chuyên môn P&ID) · 🤖 = việc code · ⛔ = chặn bước sau.

Nếu hướng này thất bại: xem `docs/BACKUP_PLAN.md` (tiêu chí thất bại, chẩn đoán, hướng thay thế).

---

## 0. Trạng thái xuất phát

| Hạng mục | Trạng thái |
|---|---|
| Stage 1 (YOLOv8, 8 coarse) | ✅ mAP50=0.995 — **không train lại**. `best.pt` nằm trên Drive, không có ở local |
| GĐ2 one-shot core (`utils/oneshot.py`, `pid-iso-prototypes`) | ✅ code xong, ✅ pytest xanh (B0, 2026-10-18) |
| `fgc.global_encoder` (prep-crops / train / infer guard) | ✅ code xong, ✅ pytest xanh (B0, 2026-10-18) |
| Per-subpart augment (OSSR-PID #2) | ✅ `SubpartWarp` (B1, 2026-10-18), user đã duyệt ảnh |
| GĐ4 one-shot inference | ❌ chưa code |
| GĐ5 eval one-shot + "Other" | ❌ chưa code |
| Leave-classes-out (`fgc.exclude_classes`) — protocol chính để chứng minh one-shot | ✅ code + test (B1b, 2026-10-18); danh sách H vẫn chờ B5 |
| `data/canonical/` local | ✅ DigitizePID gốc: 500 ảnh (400 train / 100 val), 59,498 ann, 32 class `Symbol_N` |
| `data/derived/` local | chỉ có `yolo_coarse/` — chưa có `crops_fgc/`, `tiles/` |
| Crop DigitizePID trên Drive (`crops_fgc.zip`) | ⚠️ **chỉ có 28/32 class** — phải cắt lại |

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
| **P1 — unseen-class one-shot** | `__global_lo__.pt` (train trên 32 − H class) | crop của H_in (giữ ra, map được ISO) + H_out (giữ ra, OOV) | one-shot acc trên H_in; "Other" với dương = H_out, âm = H_in (**cả hai đều unseen**) | **BẰNG CHỨNG CHÍNH** cho đóng góp #1, #2 |
| P2 — seen-class alignment | `__global__.pt` (đủ 32 class) | crop các class map được ISO | template ISO (nét vector) có khớp crop raster không | Kết quả phụ, ghi rõ KHÔNG phải one-shot; P2 − P1 = cái giá của "unseen" |
| P3 — real, unseen (nếu có) | `__global__.pt` | ASU PID_Dataset (`~/Downloads/PID_Dataset.zip`, Stage-2 test: 12 sheet thật, ~102 class) | như P1 trên bản vẽ thật | Bằng chứng mạnh nhất; chờ user quyết định giải nén + map class |
| Baseline | encoder chỉ-ImageNet (B4a) | như P1 | encoder chưa học P&ID | Cho thấy train DigitizePID đóng góp bao nhiêu |

Quy tắc bắt buộc:
- **Danh sách H chọn TRƯỚC khi train và TRƯỚC khi thấy bất kỳ kết quả nào** (B5), ghi cố định
  vào config + HANDOFF — chọn sau khi thấy số là cherry-picking.
- **Calibrate ngưỡng và báo cáo trên 2 phần tách biệt**: DigitizePID chỉ có split train/val,
  nên với P1 dùng crop H từ ảnh split `train` để calibrate `oneshot.other_min_cosine`, báo cáo
  trên crop H từ ảnh split `val` (encoder chưa thấy class H ở split nào nên cả hai đều hợp lệ).
- **Chặn rò rỉ bằng code** (B1b, B6b): eval P1 phải từ chối chạy nếu một class test nằm trong
  `fine_classes` của checkpoint.
- So với OSSR-PID **chỉ bằng số P1**, kèm câu nêu rõ khác biệt protocol.

---

## Thứ tự: CODE HẾT TRƯỚC, RỒI MỚI CHẠY (user chốt 2026-10-14)

**Phase A — Code** (🤖, local, test bằng encoder giả / model chưa train; không cần Colab):

```
(B0 ✅) ─► (B1 ✅) ─► (B1b ✅) ─► B2 Colab plumbing ─► B6a GĐ4 ─► B6b GĐ5 ─► B7-code GĐ3 ─► B4a baseline
B5 🧑 map symbol_map.csv + CHỌN danh sách H ──── song song suốt Phase A, ⛔ PHẢI XONG trước R1 ────►
```

**Phase B — Chạy từng GĐ** (🧑 Colab, theo thứ tự, mỗi bước xong mới sang bước sau):

```
(B4a baseline ImageNet) ─► R1a = train encoder LEAVE-OUT ─► R1b = train encoder ĐỦ 32 class
  ─► R2 = prototypes + separation (cả hai encoder) ─► R3 = B7 fine-tune ISO (CHỈ nếu R2 kém)
  ─► R4 = B8 eval P1 (+ P2, P3) + calibrate + chốt số
```

R1a trước R1b vì R1a là bằng chứng chính; nếu hết quota Colab thì vẫn có P1. Crop chỉ cần
cắt MỘT lần (`global_encoder=true` cắt đủ 32 class, kể cả class H — cần chúng để test).

Vì sao thứ tự này ổn: không bước code nào cần checkpoint thật — mọi code đều test được bằng
`_FakeEncoder` / ArcFaceModel chưa train (như `tests/test_oneshot.py` đã làm). B1 tự nhiên
nằm trước R1 nên không có nguy cơ train rồi mới thêm augment.

**Cái giá phải chấp nhận**:
- Code B6a/B6b chỉ được kiểm với encoder giả → lỗi ghép với model thật lộ ra ở R2/R4. Giảm
  bằng cách test end-to-end với ArcFaceModel chưa train trên crop thật (như
  `test_iso_prototypes.py` làm với 39 symbol ISO thật).
- Câu hỏi khoa học lớn nhất — encoder DigitizePID có chuyển sang ISO không — chỉ trả lời ở R2.
  Vì vậy **B7 (GĐ3) được CODE SẴN trong Phase A** dù có thể không cần chạy: nếu R2 kém thì
  chỉ việc chạy R3, không phải quay lại code giữa Phase B.
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

## B2 — Colab plumbing 🤖

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
prep --step crops   fgc.global_encoder=true                       # 1 lần, đủ 32 class
# R1a — encoder cho P1 (bằng chứng chính):
fgc --train         fgc.global_encoder=true fgc.metric_learning=true fgc.epochs=60 \
                    'fgc.exclude_classes=[<danh sách H từ B5>]'
# R1b — encoder đủ 32 class, cho pipeline end-to-end + P2 + P3:
fgc --train         fgc.global_encoder=true fgc.metric_learning=true fgc.epochs=60
```

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
| mean cao (≥ ~0.6) | Cả space co cụm — encoder không transfer sang ISO | Làm GĐ3 (B7) |

Các ngưỡng trên là **ước lượng ban đầu**, ghi lại số thật vào HANDOFF — chính các số này
là một bảng trong luận văn (separation trước/sau GĐ3).

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

**Chọn danh sách H** (đã chốt: leave-classes-out là protocol CHÍNH — xem mục "Protocol đánh
giá"). Làm ngay sau khi map xong, TRƯỚC R1, TRƯỚC khi thấy bất kỳ kết quả nào:
- **H_in — 6–8 class map được sang ISO**, trải nhiều họ (không chỉ valve: có blind_disc / HX
  nếu map được), và **ít nhất 1–2 ca khó**: class bị giữ ra mà "hàng xóm" ISO gần nó vẫn còn
  trong train (vd. giữ ra globe valve nhưng train gate valve) — đo được khả năng phân biệt
  tinh, không chỉ nhận dạng thô.
- **H_out — 2–4 class OOV** (vd. instrument bubble ISA, plug valve edrawmax) → tập "Other"
  thật sự unseen.
- Đừng rút cạn một họ: mỗi coarse family vẫn còn ít nhất 1 class trong train.
- Ghi H vào `fgc.exclude_classes` (B1b) + HANDOFF kèm lý do chọn từng class.
- (Tùy quota Colab) 2–3 bộ H khác nhau → báo cáo trung bình ± độ lệch, chắc hơn 1 bộ.

Hiện `symbol_map.csv` có 9/32 dòng DONE, 23 dòng TODO (đa số là valve và instrument).

Ngoài ra: `data/raw/real/` đang trống. Nếu có P&ID **thật** (scan/bản vẽ dự án) thì đó là
tập test cross-domain mạnh hơn DigitizePID synthetic — ghi vào HANDOFF §3 khi có.

---

## B6a — GĐ4: one-shot inference 🤖 (Phase A)

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
- **P1**: chỉ lấy crop của class H; calibrate trên crop từ ảnh split `train`, báo cáo trên
  crop từ ảnh split `val`. **Guard rò rỉ**: nếu class test nào nằm trong `fine_classes` của
  checkpoint encoder → raise, không chạy.
- **In-vocab**: top-1 / top-3 one-shot accuracy, confusion theo ISO fine_name, per-coarse.
- **"Other"** (in-vocab = negative, OOV = positive): precision, recall, F1 tại ngưỡng hiện
  tại; **AUROC** trên max-cosine (không phụ thuộc ngưỡng — con số chính của đóng góp #2).
- **Sweep ngưỡng** 0.0→1.0: in-vocab accuracy (sau reject) vs OOV recall → vẽ PNG, chọn
  ngưỡng (vd. tại false-reject in-vocab = 5%) → đây là **calibration** của
  `oneshot.other_min_cosine`, thay placeholder 0.35.
- Ablation sẵn có qua config, không cần code thêm: `aggregate=vote|mean`,
  `query_views` 1 view vs 8 view, `fgc.aug.subpart.p` 0 vs 0.5 (cần 2 lượt train).
- Ghi vào `metrics.json` dưới key `oneshot`.
- `tests/test_oneshot_metrics.py`: AUROC = 1.0 khi tách hoàn hảo, 0.5 khi ngẫu nhiên; sweep
  đơn điệu; tập OOV rỗng → bỏ qua metric "Other" có cảnh báo, không crash; **P1 với class
  test có trong checkpoint → raise**; P1 calibrate/báo cáo đúng hai split khác nhau.

---

## B7 — GĐ3: fine-tune với ISO template — CODE ở Phase A 🤖, CHẠY ở R3 🧑 (chỉ khi R2 kém)

Fine-tune `__global__.pt` trên 39 template ISO với augment mạnh (B1 subpart + `refit_box`
nhiều scale) — 1 ảnh/class nên mỗi epoch sinh N view/template. Nguy cơ chính: **quên**
không gian DigitizePID → trộn batch ISO với batch DigitizePID, LR thấp (1e-4), ít epoch.

**Code (Phase A)** — đề xuất thiết kế:
- `src/pipeline/iso_finetune.py` + entrypoint `pid-iso-finetune` + subcommand trong
  `run_pipeline.py`.
- Head ArcFace mới cỡ 32 (DigitizePID) + N (ISO) class; 32 hàng đầu copy từ head cũ, N hàng
  ISO **khởi tạo bằng prototype** của chính template đó (đã có `build_prototypes`) → bắt đầu
  từ đúng điểm one-shot, fine-tune chỉ kéo tách thêm.
- Mỗi batch trộn crop DigitizePID + view ISO (tỉ lệ cấu hình được); backbone LR 1e-4, head LR
  cao hơn; vài epoch. Lưu `__global_iso__.pt` → chạy lại `pid-iso-prototypes` với ckpt này.
- Config `oneshot.finetune.{epochs, lr_backbone, lr_head, iso_ratio, views_per_template,
  exclude_iso}`.
- **Chống rò rỉ khi fine-tune**:
  - Batch DigitizePID trong fine-tune PHẢI tôn trọng `fgc.exclude_classes` — fine-tune encoder
    leave-out bằng crop của class H là xóa sạch protocol P1.
  - Template ISO của H_in ĐƯỢC dùng (đó chính là "1 shot"), nhưng ghi rõ trong luận văn.
- **Giữ claim #3 trung thực** ("thêm symbol không cần train lại"): `exclude_iso` giữ 5–8 symbol
  ISO ra khỏi fine-tune, rồi kiểm chúng vẫn nhận đúng chỉ bằng prototype. Nếu chúng tụt mạnh
  so với symbol đã fine-tune → claim #3 chỉ đúng khi không có GĐ3, phải viết đúng như vậy.
- Test với resnet18 chưa train + vài template giả: head mới đúng kích thước, 32 hàng đầu
  không đổi trước bước optimizer đầu, checkpoint load lại được bằng `load_arcface_encoder`.

**Chạy (R3)**: chạy trên encoder leave-out trước (đó là encoder của bằng chứng chính). Kiểm
separation (R2) tăng **và** accuracy P1 không giảm **và** nhóm `exclude_iso` không tụt. Nếu R2
đã tốt thì không chạy, ghi lý do vào luận văn — code vẫn giữ làm phần ablation có sẵn.

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

1. **P1** (encoder leave-out): calibrate ngưỡng trên crop H từ split `train` → ghi vào
   `config.yaml` (`oneshot.other_min_cosine`) **kèm comment** nguồn gốc (run nào, tiêu chí
   false-reject bao nhiêu) → báo cáo trên crop H từ split `val` với ngưỡng đã chốt (không
   tune lại).
2. **P2** (encoder đủ 32 class): cùng ngưỡng, báo cáo riêng, gắn nhãn "seen-class".
3. **P3** nếu đã có ASU: cùng ngưỡng, báo cáo riêng.
4. Baseline ImageNet (B4a) chạy P1.
5. Bảng luận văn, tách theo protocol:
   - Bảng chính: P1 one-shot acc + "Other" AUROC/P/R, cạnh OSSR-PID 85.78% kèm câu nêu khác
     biệt protocol.
   - Bảng phụ: P2, P3, baseline ImageNet, ablations (vote/mean, 1 vs 8 view, subpart on/off,
     GĐ3 on/off + nhóm `exclude_iso`).
6. Cập nhật HANDOFF §0 + §6 với số thật.

---

## Checklist nhanh

**Phase A — Code**
- [x] B0 pytest + ruff xanh (207 pass, 2026-10-18)
- [x] B1 subpart augment `SubpartWarp` + user duyệt ảnh (2026-10-18)
- [x] B1b `fgc.exclude_classes` + tests chống rò rỉ (2026-10-18)
- [ ] B2 run_pipeline `prototypes` + notebook global + zip mới + assert 32 class + cell leave-out
- [ ] B6a GĐ4 one-shot infer + coarse_map
- [ ] B6b GĐ5 metrics + sweep + AUROC, theo protocol P1/P2/P3 + guard rò rỉ
- [ ] B7-code GĐ3 fine-tune ISO (code sẵn, chạy có điều kiện; tôn trọng exclude_classes + exclude_iso)
- [ ] B4a-code baseline ImageNet (tùy chọn)
- [ ] B5 🧑 hoàn thành `symbol_map.csv` (ISO / OOV) + **chọn H_in, H_out** — ⛔ trước R1

**Phase B — Chạy** (🧑 Colab, đúng thứ tự)
- [ ] (B4a) baseline ImageNet → separation "trước khi học"
- [ ] R1a = train `__global_lo__.pt` (leave-out H), copy về Drive
- [ ] R1b = train `__global__.pt` (đủ 32 class), copy về Drive
- [ ] R2 = prototypes cho cả hai encoder, ghi số separation
- [ ] R3 = B7 fine-tune ISO (chỉ nếu R2 kém; encoder leave-out trước)
- [ ] R4 = B8: P1 (chính) + P2 + P3 + baseline, calibrate trên train-split, báo cáo trên val-split

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
