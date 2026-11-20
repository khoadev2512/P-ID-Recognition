# BACKUP PLAN — nếu hướng one-shot ArcFace thất bại (lập 2026-10-14)

Đọc kèm `docs/HANDOFF.md` (bối cảnh, những gì đã bỏ) và `docs/PLAN_STAGE2_ONESHOT.md` (plan
chính, protocol P1/P2/P3). File này trả lời ba câu hỏi:

1. **Thế nào thì gọi là "thất bại"?** Định nghĩa đo được, chốt TRƯỚC khi có số (§1).
2. **Thất bại vì sao?** Chẩn đoán trước khi đổi hướng; mỗi triệu chứng dẫn tới một nhánh
   backup khác nhau (§2).
3. **Các hướng thay thế**: phân tích từng hướng (§3), xếp hạng và cây quyết định (§4–§5).

> Mọi ước lượng "khả năng thành công" / "chi phí" dưới đây là **phán đoán, chưa có thí nghiệm**.
> Các ngưỡng số là **đề xuất**, user chốt lại trước khi chạy Phase B.

---

## 0. Ràng buộc chi phối mọi lựa chọn

| Ràng buộc | Hệ quả |
|---|---|
| Deadline: HANDOFF ghi "~3 tháng" tính từ ~2026-10-09 → khoảng **đầu tháng 1/2027** (cần user xác nhận) | Hướng nào cần > 3–4 tuần code mới là quá đắt cho backup |
| GPU = Colab (~6h/session, quota) | Ưu tiên hướng tái dùng crop + encoder đã có, train ngắn |
| Luận văn thạc sĩ: không cần breakthrough (HANDOFF §1) | Một kết quả âm **được đo kỹ và giải thích được** vẫn là luận văn hợp lệ |
| 3 đóng góp: #1 one-shot, #2 "Other", #3 gắn ISO / thêm symbol không train lại | Đánh giá mỗi backup theo việc nó **giữ lại đóng góp nào** |
| HANDOFF §4: render synthetic toàn sheet **đã bỏ, đừng làm lại** | Không backup nào quay lại hướng đó (§3-D2 là biến thể khác, nói rõ ở đó) |
| Giới hạn dữ liệu test: ~17/39 symbol ISO (pump, compressor, vessel, motor, agitator, hose…) **không có crop test nào** trong DigitizePID | Đúng với MỌI hướng: claim chỉ được nói cho các họ đã đo; các họ còn lại chỉ đo được nếu có ASU / data thật |

---

## 1. Định nghĩa "thất bại" — chốt trước khi có số

So với **baseline**, không so với số tuyệt đối (OSSR-PID 85.78% dùng protocol khác, chỉ để
tham khảo). Hai baseline rẻ nhất làm mốc: encoder chỉ-ImageNet (B4a) và template matching cổ
điển (§3-B2, đề xuất thêm).

| Mã | Triệu chứng | Tiêu chí đề xuất | Đo ở |
|---|---|---|---|
| **F1** | Không transfer: prototype ISO dồn cục, one-shot gần ngẫu nhiên | P1 top-1 không vượt rõ baseline ImageNet **và** sau GĐ3 vẫn vậy | R2 + R4 |
| **F2** | Transfer một phần: được ở họ đã thấy (valve), hỏng ở họ mới / ca khó | P1 tốt trên H_in "dễ", kém trên ca khó hoặc theo họ | R4 (per-coarse) |
| **F3** | Phân loại được nhưng "Other" không tách | P1 accuracy ổn nhưng AUROC "Other" ≲ 0.70 | R4 |
| **F4** | Được trên synthetic, hỏng trên bản vẽ thật | P1 ổn, P3 (ASU) sụt mạnh | R4 (nếu có P3) |
| **F5** | Thất bại vận hành: hết quota, train không hội tụ, trễ deadline | Không ra được `__global_lo__.pt` đúng hạn | R1 |

Ghi chú: "AUROC 0.70" là con số đề xuất cho mức "có ích tối thiểu" của một bộ phát hiện —
0.5 là ngẫu nhiên. User nên chốt con số riêng trước Phase B.

---

## 2. Chẩn đoán trước khi đổi hướng

Đổi hướng mà không biết hỏng ở đâu thì dễ đổi sang một hướng hỏng y hệt. Các bước chẩn đoán
sau đều rẻ (dùng embedding đã tính ở R4), làm theo thứ tự:

| # | Kiểm | Nếu thấy… | Thì nguyên nhân có khả năng là… | Nhánh backup |
|---|---|---|---|---|
| D1 | **Top-k**: đáp án đúng có nằm trong top-3/top-5 không? | Top-1 sai nhưng top-5 thường đúng | Biểu diễn gần đúng, **cách chấm điểm/quyết định** yếu | §3-C, §3-A5 |
| | | Đáp án đúng cũng không có trong top-5 | **Biểu diễn** hỏng | §3-A, §3-B |
| D2 | **Tiền xử lý**: so hình template vs crop SAU transform (in ra cạnh nhau) | Template bị bóp méo / crop có viền & đường ống, template thì không | **Lệch đầu vào**, không phải lỗi model | §3-A4 (sửa trước mọi thứ) |
| D3 | **Theo họ**: accuracy theo coarse | Hỏng tập trung ở họ chưa thấy / ca khó | Encoder học đặc trưng **riêng của 32 class**, không tổng quát | §3-A2, §3-A3 |
| D4 | **Hình chiếu** UMAP/t-SNE của crop + prototype | Prototype ISO nằm thành một cụm riêng, xa mọi crop | **Khoảng cách phong cách** (vector sạch vs raster nhiễu) | §3-A4, §3-D2 |
| | | Prototype nằm giữa các cụm crop nhưng lẫn nhau | Biểu diễn **không đủ tinh** | §3-A3, §3-B1 |
| D5 | **Phân bố cosine** known vs unknown (histogram) | Hai phân bố chồng gần hết | Max-cosine không phải tín hiệu tốt | §3-C |
| D6 | **Nhìn tận mắt**: lưới "crop → 5 template gần nhất" cho user duyệt | User thấy lỗi hợp lý (2 symbol thật sự gần giống) | Giới hạn tự nhiên của dữ liệu, không phải bug | Báo cáo như phân tích lỗi (§3-E2) |

**Bằng chứng có sẵn cho D2** (kiểm 2026-10-14):
- `utils/dataset.py` dùng `Resize((imgsz, imgsz))` → **bóp méo tỉ lệ**. Tỉ lệ rộng/cao của 39
  template ISO trải từ **0.38 đến 3.58** (median 1.14) — vd. `pipeline` 290×101 bị nén ~3.6×
  theo chiều ngang.
- Crop DigitizePID được cắt **thêm 10% viền** (`CROP_MARGIN_FRAC`) kèm đường ống/chữ xung
  quanh; template ISO thì cắt sát bbox vector, nền trắng sạch.
- `refit_box` ở query side đã xử lý một phần lệch về bbox, nhưng chưa xử lý méo tỉ lệ.

→ Nhiều khả năng một phần "thất bại" (nếu có) đến từ đây. §3-A4 nên làm **trước** bất kỳ
backup nặng nào — và có lẽ nên làm luôn trong Phase A (xem §5).

---

## 3. Các hướng thay thế — phân tích chi tiết

Thang đánh giá: **Chi phí** (code + compute), **Khả năng giải quyết** (phán đoán), **Giữ đóng
góp nào**, **Rủi ro**.

### Nhóm A — Giữ nguyên khung (one-shot + "Other" + ISO), đổi encoder / đầu vào

Ưu điểm chung: tái dùng gần hết code (`PrototypeDirectory`, voting, gate, eval P1/P2/P3) — chỉ
cần một encoder có `.embed()` (giao diện `Encoder` trong `utils/oneshot.py`). Cả ba đóng góp
được giữ nguyên.

#### A4 — Chuẩn hóa đầu vào (canonical preprocessing) ⭐ làm đầu tiên

- **Ý tưởng**: đưa template và crop về cùng một "dạng chuẩn" trước khi vào encoder:
  1. Nhị phân hóa (Otsu).
  2. Cắt sát theo phần mực (ink bbox) → bỏ chênh lệch viền 10% giữa crop và template.
  3. **Letterbox** giữ tỉ lệ (pad trắng thành hình vuông) thay vì bóp méo.
  4. (Tùy chọn) chuẩn hóa độ dày nét: skeletonize rồi dilate về độ dày cố định — template
     vector và crop raster có độ dày nét rất khác nhau.
- **Vì sao có thể hiệu quả**: xử lý trực tiếp lệch đầu vào ở §2-D2; không đổi model.
- **Chi phí**: thấp — 1 transform + test; nhưng **phải train lại encoder** vì encoder phải
  thấy cùng phân phối đầu vào lúc train và lúc dùng.
- **Rủi ro**: cắt sát theo mực trên crop có đường ống chạy qua → bbox mực dính cả đường ống.
  Giảm bằng cách lấy connected component lớn nhất ở tâm. Skeletonize có thể làm mất chi tiết
  nhỏ (chấm đặc giữa valve) → để là tùy chọn, đo ablation.
- **Giữ**: #1 #2 #3.

#### A1 — Encoder foundation tự giám sát (vd. DINOv2), không train trên P&ID

- **Ý tưởng**: thay encoder DigitizePID bằng ViT tự giám sát đã pretrain (DINOv2 có bản trên
  timm / torch hub — kiểm tên model chính xác khi làm). Các encoder loại này nổi tiếng mạnh ở
  bài toán truy hồi / kNN mà không cần fine-tune.
- **Vì sao có thể hiệu quả**: nếu encoder DigitizePID hỏng vì **học quá khít 32 class** (D3),
  một encoder chưa từng bị ép vào 32 class có thể tổng quát hơn.
- **Chi phí**: rất thấp — không train; chỉ bọc model thành `.embed()`. Chạy CPU/Colab vài phút.
- **Rủi ro**: pretrain trên ảnh tự nhiên, bản vẽ đường nét là domain lạ → có thể kém hơn
  ImageNet-ResNet. Không biết trước — đó là lý do nên đo.
- **Giữ**: #1 #2 #3. Đóng góp "phương pháp" nhạt hơn (dùng model có sẵn), nhưng nếu kết hợp
  với A4 + gate "Other" + khung ISO thì vẫn là một pipeline của ta.
- **Ghi chú**: kể cả khi hướng chính thành công, A1 vẫn là **một dòng baseline có giá trị**
  trong bảng kết quả.

#### A3 — Train theo episode (Prototypical Networks) thay vì phân loại 32 class

- **Ý tưởng**: ArcFace tối ưu bài toán **phân loại đóng** 32 class; thứ ta cần lại là "khớp 1
  ảnh mẫu với ảnh truy vấn". Train theo **episode** mô phỏng đúng bài test: mỗi bước lấy N class,
  1 ảnh support/class, vài ảnh query; loss = phân loại query theo khoảng cách tới support.
  Có thể làm **episode chéo phong cách**: support = crop đã làm sạch (A4) / template giả lập
  (D2), query = crop nhiễu gốc.
- **Vì sao có thể hiệu quả**: mục tiêu train khớp mục tiêu test → thường tổng quát sang class
  mới tốt hơn mục tiêu phân loại. Bằng chứng gần gũi: nhóm ASU trong `PID_Dataset.zip`
  (`3__Stage-2`) đã dùng Siamese + triplet loss cho bài toán few-shot trên P&ID **thật**.
- **Chi phí**: trung bình — sampler episode + loss mới (~1 tuần code + test), 1–2 lượt train.
  Tái dùng crop, augment (B1), protocol P1, eval.
- **Rủi ro**: chỉ 32 − K class để lấy mẫu episode → ít đa dạng class, có thể vẫn overfit.
  Gộp thêm class từ ASU (D1) giảm rủi ro này.
- **Giữ**: #1 #2 #3. Thậm chí **mạnh hơn** về mặt lập luận ("train đúng cho bài toán one-shot").

#### A2 — Tiền huấn luyện tự giám sát trên crop P&ID (contrastive)

- **Ý tưởng**: học biểu diễn từ crop P&ID **không cần nhãn** (SimCLR/BYOL-style: hai augment
  của cùng một crop phải gần nhau). Nguồn: crop DigitizePID + crop / patch từ sheet ASU. Sau
  đó dùng làm encoder (có thể fine-tune nhẹ bằng ArcFace).
- **Vì sao có thể hiệu quả**: encoder học **nét, hình cơ bản** của bản vẽ kỹ thuật thay vì
  học "32 nhãn" → tổng quát sang họ chưa thấy tốt hơn (đúng chỗ F2 / D3).
- **Chi phí**: trung bình–cao — vòng train mới, cần batch lớn / nhiều epoch → tốn quota Colab.
- **Rủi ro**: cần nhiều compute nhất trong nhóm A; kết quả khó đoán.
- **Giữ**: #1 #2 #3. Xếp sau A3 vì đắt hơn và ít trực tiếp hơn.

#### A5 — Nhiều support thay vì một (one-shot → few-shot)

- **Ý tưởng**: mỗi symbol ISO có nhiều hơn 1 vector tham chiếu: nhiều view augment của
  template (đã có `template_views`), hoặc 3–5 ví dụ thật do user cắt tay từ bản vẽ thật.
- **Chi phí**: rất thấp (view augment: chỉ đổi config) → thấp (user cắt ví dụ thật).
- **Đánh đổi**: đóng góp #1 đổi tên thành **few-shot** (k nhỏ). Đóng góp #3 vẫn giữ — thêm
  symbol = thêm vài ảnh, vẫn không train lại.

### Nhóm B — Đổi hẳn dạng biểu diễn

#### B2 — Template matching / descriptor cổ điển (không học) ⭐ nên có làm baseline

- **Ý tưởng**: so khớp hình trực tiếp, không cần train: chamfer / template matching có dò xoay
  + tỉ lệ, hoặc descriptor hình dạng (shape context, Hu / Zernike moments, HOG). One-shot
  là bản chất của phương pháp; "Other" = khoảng cách tốt nhất vượt ngưỡng.
- **Vì sao đáng làm**: (1) làm **baseline không học** trong bảng kết quả — reviewer sẽ hỏi
  "phương pháp đơn giản nhất cho bao nhiêu?"; (2) là backup thật sự nếu mọi encoder đều hỏng.
- **Chi phí**: thấp — OpenCV, ~2–4 ngày; chạy CPU.
- **Rủi ro**: nhạy với nhiễu, chữ và đường ống dính vào crop; kém phân biệt chi tiết nhỏ. Kỳ
  vọng thấp hơn hướng học, nhưng làm mốc thì đủ.
- **Giữ**: #1 #2 #3 (về mặt khung), nhưng đóng góp phương pháp yếu → hợp vai baseline hơn là
  hướng chính.

#### B1 — Biểu diễn đồ thị / vector như OSSR-PID (Potrace → point cloud → DGCNN)

- **Ý tưởng**: tái hiện phương pháp OSSR-PID (đã đọc kỹ, HANDOFF §5).
- **Chi phí**: **cao** — vector hóa, lấy mẫu đường, DGCNN, train; ước tính 3–5 tuần.
- **Bằng chứng ngược**: chính Table I của OSSR-PID cho thấy thêm ResNet-34 pixel chỉ được
  85.98% so với 85.78% graph → **pixel gần ngang graph**. Lý do để đổi sang graph vì thế yếu.
- **Kết luận**: **không khuyến nghị** làm backup — đắt nhất, và chỉ tái hiện công trình đã có.

#### B3 — Phân rã thành hình cơ bản + luật ghép

- **Ý tưởng**: symbol ISO có tính ghép (valve = bow-tie + chi tiết bổ sung). Phát hiện hình
  cơ bản (tam giác, tròn, đường chéo, chấm đặc) rồi ghép theo luật / ngữ pháp.
- **Ưu**: giải thích được, thêm symbol = thêm luật.
- **Chi phí / rủi ro**: cao, giòn với nhiễu, phải viết luật tay cho từng họ. **Không khuyến
  nghị** trong 3 tháng; nhắc tới ở phần "hướng tương lai" của luận văn.

### Nhóm C — Sửa riêng phần "Other" (khi F3: phân loại được, phát hiện unknown không được)

| Hướng | Ý tưởng | Chi phí | Ghi chú |
|---|---|---|---|
| **C1. Điểm tốt hơn max-cosine** | Margin top-1 − top-2; khoảng cách Mahalanobis với covariance dùng chung (ước lượng từ crop các class đã thấy); ngưỡng **riêng theo họ** | Rất thấp — chỉ đổi hàm chấm điểm trên embedding đã có | Thử ĐẦU TIÊN khi F3; mỗi cách là một dòng ablation |
| **C2. Outlier exposure** | Khi train, thêm ảnh "không phải symbol" (vùng chữ, đường ống, nền) làm âm để đẩy unknown ra xa | Thấp–trung bình — cần cắt patch nền (có thể dùng lại `scripts/extract_bg_patches.py`) | **Không được** dùng class H_out (vi phạm P1). Chỉ dùng nền / chữ |
| **C3. Bài toán xác minh** | Siamese "cùng / khác" cho cặp (crop, template) → xác suất; "Other" = mọi xác suất < τ | Trung bình | Diễn giải xác suất rõ hơn cosine; gần với cách làm của nhóm ASU |

Giữ: cả ba đều giữ #1 #2 #3 — chỉ thay cơ chế của #2.

### Nhóm D — Đổi / bổ sung dữ liệu (khi F4 hoặc khi 32 class quá ít)

#### D1 — Train thêm bằng ASU PID_Dataset (bản vẽ thật)

- **Ý tưởng**: crop Stage-2 train của ASU (73 class, ~3,166 ảnh) + DigitizePID → encoder thấy
  cả synthetic lẫn thật, nhiều class hơn (giảm overfit, hỗ trợ A3).
- **Ràng buộc protocol**: test thật (P3) phải dùng **class ASU không có trong train** — bộ
  Stage-2 test của họ có ~102 class trên 12 sheet, cần kiểm class nào không trùng 73 class train.
- **Chi phí**: thấp–trung bình — chủ yếu là dữ liệu: giải nén, map class, đưa vào layout
  `crops_fgc`. Ảnh độ phân giải thấp (sheet 1200 px, symbol ~35 px) → cần chú ý upscale.
- **Rủi ro**: license chưa rõ (Readme không ghi) → phải trích dẫn, nên hỏi tác giả trước khi
  công bố số liệu.
- **Giữ**: #1 #2 #3, và **tăng sức thuyết phục** (có bản vẽ thật).

#### D2 — Giả lập crop từ template ISO (mức crop, KHÔNG phải mức sheet)

- **Ý tưởng**: biến template ISO sạch thành "crop giả" giống crop thật: nhiễu, rung độ dày
  nét, mờ, đoạn đường ống chạy vào, chữ/tag gần đó, lệch bbox. Dùng làm support (A5), làm
  episode chéo phong cách (A3), hoặc dữ liệu fine-tune ở GĐ3.
- **Khác với hướng render đã bỏ** (HANDOFF §4): hướng cũ ghép symbol vào **cả sheet** và hỏng
  vì không khớp tỉ lệ / bố cục với bản vẽ xung quanh. D2 chỉ làm biến dạng **một crop đứng
  riêng** → không có bài toán bố cục. ⚠️ Dù vậy, user đã chủ động bỏ hướng render, nên **cần
  user đồng ý** trước khi dùng D2.
- **Chi phí**: thấp — mở rộng augment đã có (B1 subpart + `refit_box`).
- **Giữ**: #1 #2 #3.

### Nhóm E — Định khung lại đóng góp (khi hết thời gian, hoặc mọi hướng kỹ thuật đều hỏng)

| Hướng | Nội dung luận văn | Code cần thêm | Giữ đóng góp |
|---|---|---|---|
| **E1. Open-set có giám sát trên DigitizePID** | Bỏ phần ISO one-shot. Phân loại tinh 32 class bằng ArcFace + gate "Other", đánh giá "Other" bằng leave-classes-out (H_out). Ánh xạ ISO chỉ ở mức khung + demo định tính | Gần như **không** — encoder, gate, protocol leave-out đều đã/sẽ có | #2 trọn vẹn; #1 mất; #3 thành định tính |
| **E2. Nghiên cứu thực nghiệm / kết quả âm** | "Encoder học trên P&ID synthetic chuyển sang ký hiệu chuẩn ISO tới đâu?" — đo có hệ thống (P1/P2, baseline ImageNet, A1, B2), phân tích theo họ, chỉ ra **vì sao** hỏng (§2) | Thấp — chủ yếu là phân tích + hình | Đóng góp chuyển thành **hiểu biết thực nghiệm**; cần thầy hướng dẫn đồng ý |
| **E3. Thu hẹp phạm vi** | Chỉ khẳng định trên các họ chạy được (vd. valve, fitting); các họ còn lại là hướng tương lai | Không | #1 #2 #3 trong phạm vi hẹp |

E1 là **"luận văn tối thiểu khả thi"**: gần như toàn bộ code đã có hoặc nằm sẵn trong Phase A.
Nên coi nó là lưới an toàn cuối cùng và báo cho thầy hướng dẫn biết từ sớm rằng có phương án này.

---

## 4. Xếp hạng tổng hợp

| Hướng | Chi phí | Khả năng giải quyết | Giữ #1/#2/#3 | Khuyến nghị |
|---|---|---|---|---|
| A4 chuẩn hóa đầu vào | Thấp | Cao nếu lỗi ở D2 | ✅✅✅ | **Làm luôn, trước cả khi biết có thất bại hay không** |
| B2 template matching | Thấp | Thấp–TB (vai baseline) | ✅✅✅ (yếu) | **Làm như baseline** |
| A1 DINOv2 | Rất thấp | TB, không đoán được | ✅✅✅ | **Làm như baseline**, thành hướng chính nếu thắng |
| C1 điểm "Other" tốt hơn | Rất thấp | Cao cho F3 | ✅✅✅ | Thử đầu tiên khi F3 |
| A5 few-shot support | Rất thấp | TB | #1 → few-shot | Khi top-k gần đúng (D1) |
| A3 episodic ProtoNet | TB | Cao cho F1/F2 | ✅✅✅ | **Backup chính** cho F1/F2 |
| D1 thêm data ASU | Thấp–TB | Cao cho F4, hỗ trợ A3 | ✅✅✅ | Backup chính cho F4 |
| D2 crop giả lập | Thấp | TB | ✅✅✅ | Cần user đồng ý (dính hướng render) |
| C2 outlier exposure | Thấp–TB | TB | ✅✅✅ | Sau C1 |
| C3 xác minh Siamese | TB | TB | ✅✅✅ | Sau C1, C2 |
| A2 contrastive tự giám sát | TB–Cao | TB | ✅✅✅ | Chỉ khi còn thời gian và A3 không đủ |
| E1 / E2 / E3 | Rất thấp | Luôn "xong" | Một phần | Lưới an toàn cuối |
| B1 graph như OSSR-PID | Cao | TB | ✅✅✅ | **Không khuyến nghị** |
| B3 primitive + luật | Cao | Thấp | ✅✅✅ | **Không khuyến nghị** (để làm hướng tương lai) |

---

## 5. Cây quyết định & mốc thời gian

```
R2/R4 xong ─► chạy chẩn đoán §2 (D1–D6)
   │
   ├─ D2 thấy lệch đầu vào ─────────────► A4 (nếu chưa làm) → train lại → đo lại
   │
   ├─ F3 (phân loại ổn, "Other" kém) ───► C1 → C2 → C3
   │
   ├─ F1/F2, top-5 thường đúng ─────────► C1 (đổi điểm) + A5 (nhiều support)
   ├─ F1/F2, top-5 cũng sai ────────────► A1 (rẻ, đo ngay) ─► A3 (+ D1 nếu có ASU)
   │
   ├─ F4 (synthetic ổn, thật kém) ──────► D1 → (A3 trên data trộn)
   │
   └─ F5 / hết thời gian ───────────────► E1 (+ E2 phân tích), báo thầy hướng dẫn
```

**Mốc gợi ý** (giả định deadline ~đầu 1/2027 — user xác nhận):

| Mốc | Việc | Nếu trễ mốc này… |
|---|---|---|
| Cuối tháng 10 | Xong Phase A (code), B5 (chọn H) | Bỏ B4a, B7-code khỏi Phase A; chuyển thẳng Phase B |
| Giữa tháng 11 | Có số R2 + P1 đầu tiên → **điểm quyết định chính**: tiếp tục, hay chuyển backup | Chuyển thẳng sang nhánh rẻ nhất phù hợp triệu chứng |
| Cuối tháng 11 | Hạn chót **bắt đầu** một backup tốn công (A3, D1) | Sau mốc này chỉ còn A1 / C1 / A5 / E |
| Giữa tháng 12 | **Đóng băng thí nghiệm**, chọn khung luận văn (hướng chính / backup / E1–E2) | Viết với những gì đã có |
| Tháng 12 → 1/2027 | Viết luận văn | — |

---

**Trạng thái 2026-11-16**: đã qua "điểm quyết định chính". Kết quả: F1 trên P1, F2 trên P4, F3 một phần
(τ không chuyển miền). A4 / B2 / A1 (§6) chưa từng làm. Nhánh đã chọn → `docs/PLAN_PHASE_C.md`
(Bước 1: A4 + A1 + B2 + thí nghiệm ảnh mẫu cùng phong cách, không train; Bước 2 chỉ khi đạt luật đã
chốt; E2 + E3 nếu không).

## 6. Đề xuất đưa vào plan chính NGAY BÂY GIỜ (chờ user chốt)

Ba việc dưới đây rẻ, có ích **dù hướng chính thành công hay không** (làm bảng baseline đầy đủ
hơn), và làm sớm thì tránh phải train lại:

1. **A4 chuẩn hóa đầu vào → Phase A, trước R1.** Nếu làm sau R1 thì phải train lại encoder.
   Làm sau B1 (augment), đo ablation "có / không A4" bằng cờ config.
2. **B2 template matching → baseline trong Phase A.** Code nhỏ, CPU, cho mốc "không học".
3. **A1 encoder DINOv2 → baseline cạnh B4a (ImageNet).** Gần như không tốn gì: chỉ bọc
   model thành `.embed()` rồi chạy P1.

Nếu user đồng ý, cập nhật `docs/PLAN_STAGE2_ONESHOT.md` (Phase A + checklist) cho ba mục này.
