# PLAN — Vẽ lại PID2Graph OPEN100 theo chuẩn ISO 10628-2 (lập 2026-10-28)

Nhánh thử nghiệm **song song** với `docs/PLAN_STAGE2_ONESHOT.md`, KHÔNG thay thế nó. Có cổng
go/no-go (G1) sau 2–3 ngày: không đạt → dừng, quay về one-shot như HANDOFF đã chốt.

Ký hiệu: 🧑 = việc của user · 🤖 = việc code · ⛔ = chặn bước sau · 📎 = mượn ý tưởng SynthPID
(chỉ tham khảo, **viết lại** — repo của họ không có LICENSE, xem `third_party/README_SOURCE.md`).

---

## 0. Bối cảnh và quyết định

**Vì sao thử lại hướng render** (HANDOFF §4 đã bỏ 2 cách): cả hai lần trước hỏng ở khâu **ghép**
— dán PNG raster bị co giãn nên nét không đều (tỉ lệ nét/kích thước giữa các template lệch tới
~6×), tô trắng bbox cũ, ống nối vào tâm ký hiệu, topology ngẫu nhiên. Hướng này **vẽ lại toàn bộ
bằng vector trên topology thật**, nên các lỗi đó không còn chỗ xảy ra. Rủi ro còn lại: ảnh "quá
sạch", mỗi lớp chỉ một cách vẽ, chỉ có 12 bản vẽ gốc.

**Vai trò của dữ liệu sinh ra** (🧑 xác nhận ở V0):

| Vai trò | Dùng? | Ràng buộc |
|---|---|---|
| Tập test **đầu–cuối cấp cả tờ** theo ISO (Stage 1 → one-shot → "Other") | ✅ Chính | Thiếu biến thể nét vẽ thì điểm one-shot bị thổi phồng (ảnh hỏi ≈ template) → phải có biến thể + làm suy giảm, và ghi rõ hạn chế |
| Train Stage 1 (YOLO coarse / agnostic) | ✅ Tùy chọn | Ghi rõ detector đã thấy hình dạng các lớp H |
| Train encoder ArcFace | ❌ Cấm | Phá protocol P1 (encoder thấy hàng nghìn mẫu ISO/lớp → không còn one-shot) |

**Dữ liệu đã có** (kiểm 2026-10-27): `data/raw/pid2graph/open100/` — 12 ảnh PNG (2000–3300 px)
+ 12 GraphML. Bản zip đầy đủ ở `~/Downloads/PID2Graph.zip` (còn `Dataset PID` 500 khung xương
và `PID2Graph Synthetic` 250, dùng ở V9 nếu cần).

**Định dạng GraphML OPEN100** (đã đọc thật):
- Đỉnh: `label` + bbox (`xmin, ymin, xmax, ymax`). Lớp: `connector` 2408, `crossing` 734, `arrow`
  451, `instrumentation` 350, `valve` 257, `general` 234, `inlet/outlet` 96, `background` 48,
  `tank` 35, `pump` 16.
- Cạnh: nối tâm–tâm, chỉ có `edge_label` — **cả 4567 cạnh đều `solid`** (không phân biệt tín hiệu).
- **Không có nhãn chữ.**
- `connector` (8×8 px) nằm ở chỗ ống gấp **và sát hai đầu ký hiệu** → cho đường đi thật của ống
  và vị trí cổng thật. `crossing` gồm cả chỗ bắt chéo **và ngã ba** (311 điểm bậc 3).
- Ký hiệu nhỏ: van ~20 px, general ~14 px → phải phóng tọa độ lên 3–4× khi vẽ.

---

## 1. Thay đổi so với quy trình 6 bước đề xuất trước đó

Dữ liệu thật của OPEN100 làm 5/6 bước phải chỉnh:

| # | Bản đề xuất cũ | Sửa thành | Lý do |
|---|---|---|---|
| 1 | Gộp connector thành cạnh, xóa crossing (như SynthPID) | **Giữ** connector và crossing làm hình học ống; chỉ "gộp ảo" khi đếm cổng | Gộp thì mất đường đi thật của ống và vị trí cổng; xóa crossing làm đứt 311 ngã ba |
| 1 | Bỏ cạnh tín hiệu khi đếm cổng | Lọc theo **loại đỉnh lân cận**: bỏ `instrumentation`, `background` | OPEN100 không có cạnh `non-solid`; van nối bộ truyền động qua đỉnh instrumentation (40 cạnh) |
| 2 | Dịch ngẫu nhiên từng đỉnh | **Ánh xạ trục đơn điệu** 📎 (`_build_axis_mapping`): mọi điểm cùng x dịch cùng nhau, giữ thứ tự; tương tự với y | Dịch từng đỉnh làm đoạn ống thẳng thành xiên |
| 3 | Bốc trong lớp thô + lọc cổng | Giữ, **thêm luật cho lớp không có ISO** (instrumentation → "Other", inlet/outlet, general theo kích thước) và ký hiệu có hướng | 350/1000+ ký hiệu là bong bóng ISA |
| 4 | Định tuyến Manhattan từ cổng tới cổng | **Không định tuyến lại**: giữ polyline thật, chỉ nối/cắt đoạn cuối vào cổng ISO; vẽ **cả tờ thành một SVG** rồi raster một lần | Đường thật đã vuông góc; một SVG duy nhất bảo đảm cùng độ dày w |
| 5 | Lọc trùng bằng WL hash + pHash | **Bỏ WL hash**; dùng aHash/dHash ảnh + chữ ký phép gán ký hiệu | Topology giữ nguyên → mọi biến thể cùng gốc có cùng WL hash, lọc WL sẽ loại hết |
| 6 | Làm suy giảm có giới hạn | Giữ, thêm **kiểm tra chi tiết còn sống** | Erosion/blur dễ xóa chấm/tam giác phân biệt lớp |
| + | — | **Thêm**: xuất COCO + GraphML, **chia tập theo bản vẽ gốc**, tiêu chí nghiệm thu tự động | Chống rò rỉ giữa các biến thể cùng gốc |

---

## 2. Mượn gì từ SynthPID

| Ý tưởng | File tham khảo (`third_party/synthpid/`) | Dùng thế nào |
|---|---|---|
| Đọc GraphML PID2Graph | `src/pidgen/seed_parser.py` | Viết lại gọn; giữ connector/crossing |
| Ánh xạ trục đơn điệu có jitter | `src/pidgen/perturb.py` (`_build_axis_mapping`, `_rank_remap_scene`) | Ý tưởng cho V4 (ta ánh xạ mọi tọa độ, giữ cứng ký hiệu — xem V4) |
| Chọn kiểu theo kích thước + bậc | `src/pidgen/annotate.py` (`_general_variant`) | Luật cho lớp `general` ở V3 |
| aHash / dHash tự viết bằng PIL | `src/pidgen/diversity.py` | Lọc trùng V6, không thêm dependency |
| Ý nghĩa connector/crossing | `scripts/preprocess/collapse_connectors.py` | Chỉ để hiểu dữ liệu — KHÔNG gộp |

**Không dùng**: `renderers.py` (vẽ bằng hình cơ bản, phong cách DigitizePID, nối tâm–tâm, không
cổng); các bước biến đổi topology và `apply_label_remix` (đổi lớp → nhãn sai).

## 2b. Đối chiếu với đề xuất của Gemini (2026-10-28)

Ý tưởng chung trùng với plan (vẽ trên canvas trắng, ống nối đúng cổng, làm suy giảm toàn ảnh, phân
nhóm theo cổng). Các khẳng định sai hoặc chưa kiểm, đã đối chiếu với dữ liệu/code thật:

| Gemini nói | Thực tế | Plan xử lý |
|---|---|---|
| PID2Graph có tọa độ từng đoạn ống `(x1,y1)→(x2,y2)` và tọa độ chữ | Cạnh chỉ có source/target (nối tâm–tâm); đường đi nằm ở các đỉnh `connector`; **không có nhãn chữ** | V2 dựng polyline từ connector; chữ để v2 |
| "Unified Canvas Rendering" và bộ lọc nhiễu (blur, co/giãn, JPEG) "của SynthPID" | SynthPID **không có** bước làm suy giảm (bài và code đều vậy); không có thuật ngữ này | V7 là phần ta thêm, không ghi công cho SynthPID |
| Chọn ký hiệu **chỉ theo cổng**, bỏ lớp thô | Bỏ lớp thô thì van có thể bị đặt vào chỗ một bồn chứa 2 cổng; nhóm "2 cổng thẳng" trộn van với reducer/kính quan sát vốn là `general` | Lớp thô + khớp cổng (V3) |
| Bốc **đều trong từng nhóm** → "giải quyết triệt để" mất cân bằng | Đều trong nhóm ≠ đều toàn tập: nhóm 2_angle chỉ có 2102 nên lớp này thừa mẫu | Ban đầu: bốc theo trọng số trên toàn tập (V3). Nay ⏸ hoãn, bốc đều: tập ứng viên gần như tách rời, nhóm 2_angle có 4 template sau khi sửa mã, lệch còn lại là do dữ liệu |
| Nhóm 2 cổng thẳng có van màng, van thường đóng/thường mở | ISO 10628-2 **không có** van màng; ISO không vẽ trạng thái van (xem `symbol_map.csv`, Symbol_9 và Symbol_12) | Chỉ dùng reg có trong `vocab.csv` |
| "30–40 loại van chi tiết" | Thư viện có 43 template tất cả, khoảng 10 van 2 cổng thẳng hàng | Nay `template_iso_svg/` có 62 SVG, 13 template van 2 cổng thẳng hàng (G1, 12 mã) |
| DigitizePID = "32 ký hiệu chuẩn ISA" | Trộn nhiều chuẩn (ISO + ISA + chuẩn khác) — HANDOFF §4d | — |
| Canvas cùng kích thước gốc, ví dụ 4500×7000 | Ảnh OPEN100 chỉ 2000–3300 px, van ~20 px | Phóng tọa độ ×3 (V4) |
| Vẽ bằng `cv2.line` lên ảnh có dán template → nét "trùng khớp 100%" | Template PNG co giãn thì nét co giãn theo — đúng lỗi của hai lần trước | Ghép cả tờ thành một SVG, chỉnh stroke theo scale (V5) |
| Ghép xong thì mô hình "hoàn toàn không phân biệt được" | Hết vết ghép, nhưng vẫn còn khoảng cách phong cách với bản scan thật | Ghi là hạn chế (§Rủi ro) |

Ý đáng cân nhắc nhưng **ngoài phạm vi hiện tại** (đổi vocab thì kéo theo B5 và protocol P1):
ký hiệu đầu cuối (phễu, ống thông hơi, xả đáy, nắp bịt) và ký hiệu thiết bị đo theo ISO 14617 cho
nhóm 1 cổng. Hiện plan giữ bong bóng ISA làm `other_isa` để phục vụ đóng góp "Other".

---

## 3. Bố cục code

Theo quy ước repo (flat `src/pipeline`, `src/utils`, Hydra, pytest):

```
src/utils/pid2graph.py         V2  đọc GraphML, dựng polyline ống, tính cổng của ký hiệu
src/utils/iso_templates.py     V1  nạp template SVG + ports.csv, biến đổi (xoay/scale) template
src/utils/iso_mapping.py       V3  chọn template cho từng nút: assign_symbols(...) -> Assignment
src/utils/iso_layout.py        V4  ánh xạ trục đơn điệu ×S: transform_layout(...) -> (Sheet, runs)
src/utils/page_svg.py          V5  ghép cả tờ thành SVG, raster bằng PyMuPDF (đã có trong deps)
scripts/check_g1.py            V5  đo 3 tiêu chí tự động của G1 trên ảnh 3
src/pipeline/render_iso.py     V5–V8  stage `pid-render-iso` (Hydra), gọi V2–V3, sinh dataset
configs/render_iso.yaml            luật ánh xạ, w, scale, ngân sách biến thể, tham số suy giảm
docs/symbol_mapping/iso_taxonomy.csv  V1  danh sách 62 ký hiệu: nhóm, fine_name, cổng dự kiến, placement
scripts/extract_iso_svg_templates.py  V1  script MỚI xuất SVG + ports.csv (không sửa script cũ)
data/raw/template_iso_svg/          V1  <file>.svg + ports.csv + templates.csv (gitignore, sinh lại bằng script)
data/derived/iso_open100/          V8  images/, annotations.coco.json, graphml/, splits.json
tests/test_extract_iso_svg_templates.py, test_pid2graph.py, test_iso_templates.py (đã có),
test_iso_mapping.py, test_iso_layout.py, test_page_svg.py, test_render_iso.py
```

Không thêm dependency mới (PyMuPDF, PIL, OpenCV đã khai báo). networkx chỉ có mặt gián tiếp qua
torch, **không** khai báo trong `pyproject.toml` → không import.

### Cô lập với Stage 1 / Stage 2 (bắt buộc)

Nhánh này **chỉ thêm file mới**. Những thứ Stage 1/2 đang đọc thì không được sửa và không được ghi đè:

| Thứ Stage 1/2 đang dùng | Ai dùng | Quy tắc |
|---|---|---|
| `data/raw/iso_symbols_clean/*.png` + `symbols.csv` | `oneshot.symbols_dir` → `pid-iso-prototypes` | **Không ghi vào.** Script cũ `extract_iso_symbols_svg.py` mặc định `--out` vào đúng thư mục này → KHÔNG chạy lại, KHÔNG sửa; V1 viết script mới, chỉ ghi `data/raw/template_iso_svg/` |
| `data/raw/iso_symbols/vocab.csv` | `oneshot.vocab_csv`, build classes | **Không sửa.** Nhãn `other_isa` / `other_generic` chỉ nằm trong `configs/render_iso.yaml` và COCO của dataset mới |
| `data/canonical/` (DigitizePID) | mọi stage qua `DataPaths` | **Không ghi vào.** Dataset mới ở thư mục riêng (V8) |
| `configs/config.yaml` | mọi entrypoint Hydra | **Không sửa.** Stage mới có file config riêng `configs/render_iso.yaml` |
| `src/pipeline/*`, `src/utils/*` hiện có | Stage 1/2 | Không sửa; chỉ thêm module mới |
| `scripts/render_pid.py` (`degrade()`) | không ai (hướng đã bỏ) | **Chép** hàm sang module mới, không import — script cũ có thể bị xóa |
| `pyproject.toml` | — | Chỉ **thêm** một dòng entry `pid-render-iso` |

Kiểm sau mỗi bước V: `uv run pytest` toàn bộ phải xanh như trước (test cũ không đổi), và
`git status` không có thay đổi nào trong các đường dẫn ở bảng trên ngoài `pyproject.toml`.

Git: làm trên branch riêng `feat/iso-render-pid2graph`, tách từ branch Stage 2 **sau khi** commit
phần B5 đang dở (vocab, symbol_map, iso_prototypes, check_b5…) để hai việc không lẫn commit.

---

## V0 — Chốt vai trò + vệ sinh dữ liệu 🧑 ⛔

- [ ] Xác nhận vai trò ở bảng §0 (test đầu–cuối; có train Stage 1 hay không) — user quyết sau,
      **không chặn V1–V5** (G1 chỉ cần ảnh để duyệt); chặn V8.
      Kèm quyết định tùy chọn: có train baseline có giám sát (chỉ lớp đã thấy) làm cận trên cho
      one-shot không (xem "Trọng số bốc ký hiệu" ở V3).
- [x] `PID2Graph.zip` ở gốc repo đã được gitignore (`/PID2Graph.zip`, 2026-10-28).
- [x] Ký hiệu ngoài vocab: chọn **(a)** `other_iso` (2026-10-29); phạm vi vẽ `scope: html` (45 template; 46 từ 2026-11-01, thêm 511 `flanged_connection`).
- [x] Bảng ánh xạ V3 — bản 1 duyệt 2026-10-28; **bản 2** (5 nhóm, mã đã sửa theo BK-2020) duyệt
      2026-10-29 cùng preview template, (a) và `scope: html`.
- [x] Ghi một dòng vào HANDOFF §4: "mở lại nhánh render dạng vẽ lại vector trên topology
      OPEN100, có cổng G1" — **chỉ sau khi G1 đạt**. Đã ghi 2026-11-02 (§4e).

## V1 — Template ISO dạng vector + khai báo cổng 🤖 (~1.5 ngày) — 🟢 XONG 2026-10-30

**Nguồn dữ liệu chuẩn cho mọi bước sau** (V3, V5, V7 chỉ đọc ba thứ này, không đọc
`iso_symbols_clean/` hay `vocab.csv`):

| File | Nội dung | Ai đọc |
|---|---|---|
| `docs/symbol_mapping/iso_taxonomy.csv` (tracked) | 62 dòng `reg_number, variant`: `group` G1–G5, `fine_name`, `in_vocab`, `scope` (html/extra), `html_code`, `port_config` dự kiến, `directional`, `placement` (pipe / pipe_or_end / end / attach), `port_override` | V3 (ứng viên theo nhóm), V5 (nhãn COCO) |
| `data/raw/template_iso_svg/<file>.svg` | Thân ký hiệu, đã bỏ stub, viewBox cắt sát theo mm của tờ BK-2020. `<file>` = `reg`, hoặc `reg_variant` khi một REG# có nhiều hình (X8075_1, X8075_2) | V5 (nhúng vào SVG cả tờ), V7 (kiểm chi tiết còn sống) |
| `data/raw/template_iso_svg/ports.csv` | `file, port_id, side, x_mm, y_mm, u, v, source`; x/y tính từ gốc viewBox, `u,v` = tỉ lệ 0–1; cổng nằm **trên mực của thân**; `source` = `stub` (đo), `override`, hoặc thêm hậu tố `+nearest` (chỉ X411 L) | V3 (khớp cổng), V5 (điểm ống dừng) |
| `data/raw/template_iso_svg/templates.csv` | `expected` vs `detected`, `sides`, `gap_mm` (khoảng hở cổng–mực lớn nhất), `ok` (khớp cấu hình **và** `gap_mm` ≤ 0.15), `scope`, `width_mm, height_mm` | V5 (tỉ lệ vẽ), kiểm tra |

Sinh lại (thư mục `data/raw/` bị gitignore): clone BK-2020-04 vào `third_party/bk2020` rồi
`uv run python scripts/extract_iso_svg_templates.py --preview docs/experiments/iso_render/template_iso_svg_preview.png` (~35 s).

Đã làm:
- [x] 🤖 Script mới `scripts/extract_iso_svg_templates.py`, dùng lại `isolate`, `is_stub`, `_ink_box`
  của script cũ ở chế độ chỉ đọc; từ chối ghi vào `iso_symbols_clean/`.
- [x] 🤖 Cổng đo từ stub: stub được nhận khi ≥ 25% ink nằm ngoài box thân (X8087: mũi tên chéo trùm
  stub trái); stub vẽ trùng gộp trong 1 mm (C0093). Ký hiệu không có stub (511, C0096–C0098,
  2044/2045, 2040, 2039) lấy cổng từ `port_override` (giữa cạnh box thân; riêng C0096–C0098,
  2044/2045 dùng `L,R@tail`: cổng L/R đặt tại điểm cuối của cán thay vì giữa bbox).
- [x] 🤖 **Đoạn ống thừa của ký hiệu không stub (2026-11-01)**: 511 vẫn mang hai đoạn ống 5 mm dừng
  cách vạch mặt bích 2.5 mm (cùng artefact BK-2020). `find_ports` không gỡ được vì cả 4 path của
  511 đều dài đúng 5 mm như stub. `pipe_pieces` gỡ các đoạn dài như stub, nằm **dọc trục cổng
  override** và **hoàn toàn ngoài** phần còn lại của hình. 511: rộng 18.6 → 3.7 mm (trước đó ký
  hiệu bị thu nhỏ ~7× khi vẽ), cổng ở mặt ngoài hai vạch, gap 0.024 mm. 61 template còn lại không
  đổi một byte (đã diff).
- [x] 🤖 **241 `flow_direction` thành template (2026-11-02)**: trong BK-2020 (SHT6), 241 là một đoạn
  ống 6.25 mm gắn `marker-end`: một tam giác đặc khoảng 2.36 × 1.18 mm (tỉ lệ 2:1), tâm đặt ở đầu
  đoạn ống. Taxonomy có dòng mới: nhóm `G0` (không luật nào bốc), `placement: overlay`, `L,R`.
  `keep_markers_only` giữ marker đã bake và bỏ đoạn ống, vì đoạn ống chính là ống trên trang. Template
  là đầu mũi tên 2.2 × 1.19 mm, cổng L ở đáy, R ở đỉnh, gap 0.024 mm. 62 template khác không đổi
  (đã diff). Tổng: 63 template, 47 trong scope `html` (51 từ 2026-11-05: thêm X8100 và 3 agitator, V5b).
  Lý do chọn template thay vì tự vẽ: mũi tên được nhận diện như một lớp (`flow_direction`, như
  PID2Graph và Moon và cộng sự 2021), nên ảnh vẽ phải trùng ảnh support (`iso_symbols_clean/241.png`)
  và đi cùng đường với mọi lớp khác.
- [x] 🤖 **Đóng khe cổng–thân (sửa 2026-10-30)**: BK-2020 vẽ stub **cách thân 2.5 mm** (2101: thân
  x 62.5–72.5, stub dừng ở 60 và 75), nên đầu trong của stub không phải chỗ ống dừng. Trước khi
  sửa, 56/62 template hở 2.1–2.6 mm (2044/2045, X2125 B hở ~1 mm) → tiêu chí G1 số 1 chắc chắn
  trượt. Cổng `override` của đĩa có cán (C0096–C0098, 2044/2045) cũng hở 2.5 mm vì box rộng bằng
  đĩa. Nay `snap_to_body` trượt **mọi cổng** dọc trục ống vào tới mực đầu tiên của thân (raster
  1200 dpi, dải ±0.1 mm, tối đa 5 mm); trục ống trượt qua thân thì lấy mực gần nhất (`+nearest`,
  chỉ X411 L: đầu sóng lệch 0.25 mm). Kết quả: `gap_mm` lớn nhất 0.098 mm, 62/62 `ok`. Đã xem
  ảnh kiểm từng ca (van, van góc, 3 ngả, HX, đĩa, ống mềm, phễu).
- [x] 🤖 62/62 template: `detected` khớp `expected` và `gap_mm` ≤ 0.15. 19 test trong
  `tests/test_extract_iso_svg_templates.py`; `test_real_ports_touch_the_body` (loader) nay chạy
  cho cả 6 template thật, kể cả 2040, và không còn `xfail`.
- [x] 🤖 **Căn thẳng cặp cổng đối diện (sửa 2026-10-31)**: BK-2020 vẽ một stub của nhiều van lệch
  0.125 mm so với stub kia (X8071: y −95.5 và −95.625); đo ở 150 dpi (1 px = 0.169 mm) làm tròn
  thành 0 hoặc 1 px. X411 L lệch 0.22 mm do bước `+nearest`. Tổng cộng 14 cặp lệch < 1 mm, kể cả
  cặp L–R của van 3 ngả. Nay `straighten_pairs` (sau `snap_to_body`): cặp L–R (T–B) lệch ≤ 0.3 mm
  được đặt về trung bình, rồi trượt lại vào thân dọc trục mới. Lệch lớn hơn là có chủ ý (đầu nối
  của HX 2511) nên giữ nguyên. Kết quả: 0 cặp lệch, 28 cổng dịch tối đa 0.13 mm, 62/62 `ok`.
  V5 không cần xử lý riêng; loader không cần sửa.
- [x] 🧑 Duyệt preview `docs/experiments/iso_render/template_iso_svg_preview.png` (2026-10-28).
- [x] 🧑 Xem lại preview đã sinh lại 2026-10-30 (chấm đỏ trên thân, X8087 có đầu mũi tên) — ổn.

Loader `src/utils/iso_templates.py` (user viết, review 2026-10-30; 30 test trong
`tests/test_iso_templates.py`). Các quyết định đã chốt 2026-10-30, dựa trên số đo của 62 SVG:
- [x] 🤖 **Loader**: đọc 3 file trên → `Template(file, fine_name, group, ports, w_mm, h_mm, ...)`.
  `Template.body(stroke=W)` trả nội dung thân đã làm sạch:
  - Bỏ `<defs>`, metadata, `sodipodi:namedview`, và **xóa mọi `id`** trong thân. Thân không tham
    chiếu tới defs (marker duy nhất của X8087 đã được nhúng thành hình ở bước xuất, xem dưới), nên
    xóa id là đủ để nhúng hàng trăm bản sao vào một SVG mà không trùng; không cần gắn prefix.
  - `viewBox` mang tọa độ tờ gốc (vd. 2101: `59.444 23.376 …`), còn `ports.csv` tính từ gốc
    viewBox → ma trận của template phải gồm `translate(-vx, -vy)`.
- [x] 🤖 **Chuẩn hóa nét**: mỗi template lấy nét phổ biến nhất làm `w_base`; `body(stroke=W)` ghi lại
  mọi nét thành `w / w_base × W`, nét 0 giữ nguyên. **Không** bỏ stroke-width, vì 6 file trộn độ
  dày có chủ ý: X2124/X2125 có vạch 1 so với nét 0.25 (4×), C0096 có chấm 1, C0082 và X8174 trộn
  0.25/0.5, X8076 có hình chỉ tô (nét 0). Transform trong thân chỉ có translate/rotate/scale ±1,
  nên không làm méo nét. **Màu**: stroke luôn `#000000`, fill là `#000000` (20 file: chấm van cầu,
  đĩa 2044…) hoặc `none` → ảnh xám mực đen không cần `currentColor`. Chỉ đổi khi cần tô màu để
  debug, và khi đó đổi cả stroke lẫn fill.
- [x] 🤖 **Xoay/lật**: `Template.oriented(rot, flip)` trả Template mới, SVG gốc giữ nguyên: cổng đã
  cập nhật (`x_mm, y_mm, u, v, side`), `w/h` hoán đổi khi quay 90°/270°, và ma trận affine 2×3 cho
  V5. Quy ước: lật ngang trước, rồi quay `rot ∈ {0, 90, 180, 270}` thuận chiều kim đồng hồ trên màn
  hình (= `rotate(+θ)` vì trục y của SVG hướng xuống); ma trận dời lại để gốc mới ở góc trên trái.
  Quay 90° đổi cạnh L→T→R→B→L; lật đổi L↔R. **Không lọc hướng trùng**: so cổng không đủ (van một
  chiều lật vẫn có cổng {L, R} nhưng hình đảo chiều), so nét vẽ thì phải raster. Đủ 8 hướng chỉ làm
  tần suất lệch nhẹ.
- [x] 🤖 **Ràng buộc hướng**: mọi ký hiệu, kể cả `directional`, được đủ 8 hướng, vì OPEN100 không có
  chiều dòng chảy. Chọn hướng khớp trục cổng là việc của V3. Loader **không** tính `size_class`:
  `size_class` là của **đỉnh PID2Graph** (ngưỡng px, xem V3); kích thước template
  (`width_mm × height_mm`) chỉ dùng để tính tỉ lệ vẽ ở V5, còn nhóm đã có ở cột `group`.
- [x] 🤖 **X8087 mất đầu mũi tên (sửa 2026-10-30)**: đường chéo của van điều khiển dùng
  `marker-end`, mà MuPDF bỏ qua marker → render chỉ ra nơ van + một đường chéo, mất đúng chi tiết
  phân biệt lớp `valve_control`. Script xuất nay nhúng marker thành hình (`bake_markers`: chép nội
  dung marker vào `<g transform>` tại đầu đường, theo `orient`/`markerUnits`), chạy sau khi đo cổng
  nên cổng không đổi. X8087 là ký hiệu duy nhất trong 62 template dùng marker.

Ghi chú cho Stage 2 (không sửa ở đây): `iso_symbols_clean/X8087.png` vẫn còn vạch stub thừa bên
trái — bản SVG mới đã sạch; nếu cần thì sửa trong phạm vi B5.

## V2 — Đọc GraphML + dựng khung xương 🤖 (~1 ngày) — 🟢 XONG 2026-10-29

Code: `src/utils/pid2graph.py` (không dùng networkx — chưa khai báo trong `pyproject.toml`),
`tests/test_pid2graph.py` (20 test). `2_same` tách khỏi `2_angle` (2026-10-29): 16 ca trên 12 ảnh — bơm 2 (cả hai cổng ở cạnh trên, hợp lệ), tank 1, instrumentation 13 (liên kết đôi tới cùng một van, cách nhau 0–9 px); V3 không được ghép `2_same` với van góc, `scripts/check_pid2graph.py` (bảng + ảnh chồng cổng, ra
`docs/experiments/iso_render/v2_ports/`). Kết quả 12 ảnh: 2159 ống, 0 đoạn chéo; 278 ống phải gấp chữ L
(đa số là đường chéo/cong thật: crossing–crossing trực tiếp, arrow–crossing, van–bộ truyền động). Van:
217/257 là `2_straight` (H 128, V 89).

Quyết định thực thi: (1) cổng trượt dọc mép ≤ 2 px để khớp ống; (2) **cạnh của cổng**: connector nằm trên mép
(≤ 2 px) thì connector quyết định cạnh; connector nằm sâu trong bbox (bồn, `general` lớn — 29 đầu ống trên
12 ảnh) thì cạnh = **chỗ polyline thực sự cắt bbox**, cổng = giao điểm đó, đoạn nằm trong bbox bị bỏ
(vd. `general2` tờ 5: L → T; `tank1` tờ 6: T → R); (3) ống không bao giờ ra khỏi bbox (đầu kia là ký hiệu
lồng bên trong, hoặc ống bồn nối vòng về chính nó) → `PipeRun.internal = True`, không sinh cổng cho cả
hai đầu; **V3 và V5 phải bỏ qua run `internal`** (13 run trên 12 ảnh, đo lại 2026-10-29 sau khi tách
vòng nozzle — con số 25 trước đó gồm cả 12 vòng ở (4)), kèm các ký hiệu lồng chỉ nối qua chúng (luật ở V3); (4) **vòng nozzle dài 0 px** (`tank → connector → connector → tank`, 12 vòng ở tờ 2, 5, 6, 7) không phải ống: bỏ vòng, và nếu có `general` nhỏ bậc ≤ 1 trong 25 px (mặt bích nozzle) thì thay bằng một run `nozzle` dài 0 nối bồn với mặt bích — cổng bồn = chỗ tia từ vòng qua mặt bích cắt bbox, mặt bích đặt tại điểm đó (V5 phải dời mặt bích về cổng). 11/12 vòng có mặt bích; vòng còn lại (`general2` tờ 5, một `general` lớn) không có nên bị bỏ.

## V3 — Luật ánh xạ sang ISO 🤖 (~0.5 ngày) — 🟢 code sơ bộ XONG 2026-10-31, cập nhật 2026-11-01 (30 test)

`configs/render_iso.yaml`, khóa = `(label PID2Graph, size_class)` → danh sách **nhóm** (G1–G5)
ứng viên; danh sách ký hiệu trong nhóm lấy từ `docs/symbol_mapping/iso_taxonomy.csv` (cột `group`,
`placement`), không chép lại vào yaml. **Khớp cổng tính tự động** từ
`data/raw/template_iso_svg/ports.csv` (V1), theo các luật "Chọn template và hướng" bên dưới.
Bảng dưới đây là bản đọc được của taxonomy CSV —
khi lệch nhau thì CSV là căn cứ.

**Mô hình cổng (2026-10-29)** — mỗi ký hiệu khai báo **điểm nối ống** trên chính hình vẽ của nó
(vị trí và cạnh hướng ra), và ống phải dừng đúng tại đó. Mô hình có ba phần khớp nhau:
(1) cổng của template, đo từ stub BK-2020 vào `ports.csv` (V1; vd. 2101: L (0.5, 2.96) mm,
R (15.4, 2.96) mm); (2) cổng của nút PID2Graph, lấy từ chỗ connector chạm bbox (`symbol_ports`, V2);
(3) V3 chọn template có cấu hình cổng khớp với nút, rồi V5 xoay/phóng/đặt template sao cho cổng
trùng đầu ống và cắt ống tại đó.

- **SynthPID không có phần này**: `render_png` vẽ **toàn bộ ống trước** (nối tâm–tâm), **rồi vẽ ký
  hiệu đè lên**. Van là hai tam giác chỉ có viền, không tô nền, nên ống chạy xuyên qua giữa. Với họ như
  vậy là đủ, vì nhãn là **đồ thị** (Relationformer), ký hiệu là hình PIL co giãn theo bbox bất kỳ, và
  việc chèn nút vào giữa cạnh (perturb) không phải kiểm tra gì.
- **Vì sao ta cần**: (a) chi tiết phân biệt lớp nằm ngay giữa ký hiệu (chấm đặc của van cầu, vòng tròn
  của van bi, vạch giữa của van cổng). Ống xuyên tâm sẽ đè lên đúng chi tiết Stage 2 dựa vào để phân
  biệt. (b) Template ISO có hình học cố định: cổng đẩy bơm nằm tiếp tuyến, van góc có cổng dưới + trái,
  nên không nối từ tâm được. (c) Hai renderer trước đã hỏng đúng ở chỗ ống xuyên ký hiệu (HANDOFF §4),
  và tiêu chí G1 số 1 (V5: mọi cổng có ink cả hai phía trong 1 px — không hở) chỉ đo được khi có
  tọa độ cổng.
- **Khác biệt cốt lõi**: SynthPID cần ảnh đủ tốt cho **bài toán đồ thị**. Ta cần ảnh đúng tới **từng
  chi tiết ký hiệu**, vì Stage 2 phân loại chính các chi tiết đó.

**3 nguyên tắc khi số cổng của template và của nút lệch nhau** (suy ra từ mô hình cổng; sửa câu
chữ 2026-10-31):
1. **Nút ít cổng hơn template: được, nếu `placement` cho để hở.** Chỉ chọn khi `placement` là
   `pipe_or_end` hoặc `end` (G5), và mọi cạnh cổng của nút phải trùng một cạnh cổng của template; cổng
   thừa của template để hở (vd. thiết bị G5 ở cuối ống, cổng còn lại thông ra khí quyển).
2. **Nút nhiều cổng hơn template: không chọn**, vì có ống không có chỗ nối.
3. **Mũi tên vẽ đè lên ống**: 241 `flow_direction` không có cổng; nó là lớp phủ đặt dọc đoạn ống và
   xoay theo trục ống, ống không bị cắt.

Còn lại: cạnh cổng phải **khớp chính xác** sau xoay/lật. Hệ quả: 18 van 0/1 cổng (~7% của 257 van)
không có ứng viên → `other_generic`, ghi log số lượng. Cổng G1 không bị ảnh hưởng (chỉ thay van
`2_straight`).

**Chọn template và hướng (2026-10-31)** — trong tập ứng viên của nút (nhóm theo khóa + `scope` +
3 nguyên tắc trên), bốc **đều một lớp** (`fine_name`), rồi đều một template của lớp đó, rồi đều một
hướng trong các hướng hợp lệ (8 hướng của `all_orientations()`, không gộp trùng). Bốc theo lớp trước
vì Stage 2 đánh giá theo lớp: nếu bốc theo template, `valve_butterfly` (X8075 có 2 hình) sẽ được 2/13
so với 1/13 của các lớp G1 khác. Hướng hợp lệ:
- **Mặc định**: hướng mà cạnh cổng khớp chính xác (hoặc chứa cạnh của nút, theo nguyên tắc 1).
- **Bơm, máy nén**: nút `pump` chỉ nhận template `port_config = 2` (G4, 2 cổng LR); nút 2 cổng
  `2_straight`, `2_angle` hoặc `2_same` lấy các hướng có **nhiều cạnh cổng trùng nút nhất**. Đoạn gấp
  chữ L do V5 vẽ. Không để cả G4, vì như vậy bồn 2 cổng hay `general` lớn có thể thành bơm.
- **`multi`** (2062 và các bình): nhận nút có ≥ 1 cổng; hướng cố định rot 0, không lật. Cổng đặt ở
  đâu trên viền là việc của V5. Nút `tank` **chỉ** nhận `multi` (chạy thử: bồn 4 cổng từng bị gán HX
  2515 — sai ngữ nghĩa); `general` lớn nhận `multi` hoặc `4` (HX 2511/2515), vì HX trong PID2Graph
  thường gán nhãn `general`.
- **Mũi tên `flow_direction` và cờ inlet/outlet**: không có template cổng; V3 trả **trục** (H/V)
  theo cạnh của một cổng bốc ngẫu nhiên (L/R → H, T/B → V; áp cho cả nút 1 cổng, `2_angle`, 3 cổng),
  và **chiều** (`forward`) bốc ngẫu nhiên vì OPEN100 không có chiều dòng chảy. Bốc chiều ngay ở V3 để
  mọi lần bốc đi qua một `rng` → cả lần sinh tái lập được bằng seed. Nút không có cổng nào (2 nút):
  trục theo bbox (rộng ≥ cao → H, ngược lại V). Mũi tên mang `template` = 241 (`overlay`) xoay sao
  cho đỉnh (cổng R của bản gốc) trỏ theo `axis`/`forward`; không có 241 trong scope thì `None`, và V5
  vẽ tam giác dự phòng (2026-11-02).
  **Chiều theo bản vẽ, không bốc ngẫu nhiên, khi mũi tên áp sát một vật** (2026-11-02,
  `_tip_side`): trong OPEN100, 264/362 mũi tên `2_straight` có một phía mà run dài không quá một cỡ
  mũi tên (241 ca là chỗ gấp hoặc ngã ba, 11 `general`, 4 van). Đo trên pixel gốc (đầu hẹp là đỉnh,
  đã trừ độ dày ống): **257/262 trỏ vào phía đó** (98%). Vậy với các mũi tên này, `axis`/`forward` lấy
  theo phía đó. Mũi tên giữa ống (98 ca) vẫn bốc ngẫu nhiên. `rng` vẫn được rút như cũ, nên dãy ngẫu
  nhiên không bị dịch.
  **Mũi tên chỉ dẫn** (chốt 2026-11-02): 72 mũi tên 1 cổng có cạnh tới một `crossing` là **mũi tên
  chỉ dẫn**, không phải chiều dòng chảy. Đó là một đường chéo mảnh từ chữ (tag "V-1…"), đầu mũi tên
  chạm ống chính; cạnh đồ thị chính là đường đó (đầu mũi tên → `crossing` ở đầu kia). Luật
  `{label: arrow, ports: ["1"], next_to: [crossing], kind: leader}` gán kind `leader`, nhãn
  **`leader_arrow`** riêng. Lý do chọn nhãn riêng: PID2Graph và SynthPID gộp cả hai loại vào một lớp
  `arrow`, còn Moon và cộng sự chỉ có "flow arrows". Giữ hai nhãn tách thì có thể gộp lại khi so với
  PID2Graph, hoặc bỏ qua / coi là nền khi chỉ dùng `flow_direction` (ISO 241). 7 mũi tên 1 cổng còn
  lại (cạnh tới van/`general`) là mũi tên dòng chảy: run ngắn thì đỉnh trỏ vào ký hiệu, run dài
  (đầu ống cụt) thì đỉnh trỏ ra khỏi ống (5/7 theo quy luật này). Còn mở: 5 mũi tên `2_angle`
  (nằm ngay chỗ gấp) chưa có quy luật riêng.

**Nút đặc biệt**:
- `connector`, `crossing`, `background`: bỏ qua, không phải ký hiệu.
- Ký hiệu lồng: tâm nằm trong bbox của một ký hiệu khác **và** mọi run của nó đều `internal` (nút
  không có run nào cũng tính) → `kind = nested`, không vẽ, không nhãn. Trên 12 tờ: 3 nút (`arrow87`,
  `arrow99` tờ 1; `general100` 0 cổng trong bồn tờ 11). 4 arrow khác nằm trong bbox dàn ngưng ACC
  tờ 1 nhưng có ống ra ngoài nên **vẫn vẽ** (xem ghi chú ở V5).
- **Phụ kiện ở vòi bình** (sửa 2026-11-01; bản trước ghi "không cần luật riêng"). Trong OPEN100, phần
  lớn `general` nhỏ `2_straight` có ống dẫn tới `tank` là **cặp mặt bích vòi bình** (`⊣⊢`, ghi N1,
  N2…). Đã soát bằng ảnh gốc. Luật G3 chung từng bốc mọi lớp G3 cho chúng: **76/138** phụ kiện G3 trên
  12 tờ đứng ngay vòi bình, trong đó có 11 orifice plate, 14 strainer và 8 kính quan sát. Thực tế
  không ai đặt chúng ở đó: orifice plate cần đoạn ống thẳng trước/sau và đi kèm FE/FT; strainer
  thường ở ống hút bơm; kính quan sát nằm trên đường ống.
  Luật mới đứng **trước** luật G3 chung:
  `general` nhỏ `2_straight` có `next_to: [tank]` → chỉ các lớp `classes: [flanged_connection,
  blind_disc, open_disc, interchangeable_disc(_open), compensator, reducer]`. Cặp mặt bích trơn
  (511) khớp bản gốc nhất; đĩa mù và spectacle blind ở vòi bình (cách ly khi vào bình), compensator
  và reducer ở vòi thiết bị là cách lắp phổ biến. ~~Nút 1 cổng ở vòi bình vẫn đi luật G5 (van thở,
  đĩa nổ — đúng thực tế).~~ **Sai, sửa 2026-11-05** (V5b, "Sửa sau đối chiếu thực tế"): nút 1 cổng là
  cặp mặt bích ở đầu ống (53/55), giờ vẽ 511. **511 vào scope (chốt 2026-11-01)**: taxonomy `scope: html` (46
  template), được nhận diện như một lớp (`flanged_connection`, trong vocab). Bốc đều trong 7 lớp
  (trọng số vẫn hoãn), và luật G3 chung cũng có thể chọn 511 trên đường ống.
- **Đĩa và tấm giữa hai mặt bích** (2026-11-01): orifice plate, blind/open disc, interchangeable
  disc luôn được kẹp giữa hai mặt bích, và hay (không bắt buộc) được vẽ thành `──┤┴├──`. V3 tung
  `flanges.p` (0.5) cho mỗi ký hiệu thuộc `flanges.classes`; trúng thì `Assignment.flanges` = template
  `flanges.symbol` (511) xoay theo trục ống của đĩa (cùng tập cạnh cổng). 511 không trong scope →
  `None`. V5 vẽ (xem V5). **Hai nhãn riêng** (chốt 2026-11-01): đĩa và `flanged_connection` mỗi cái
  một bbox.
  Ống nối vào **đuôi cán** của đĩa, không vào giữa bbox: V1 `port_override = L,R@tail`.
- Agitator (`attach`) **không bao giờ được chọn** ở V3, vì nó không nằm trên ống. Agitator thuộc
  phần vẽ bình `multi` (xem V5).

**Hình dạng code (2026-10-31)**: `src/utils/iso_mapping.py` có
`assign_symbols(sheet, runs, templates, rng, cfg) -> dict[node_id, Assignment]`. `Assignment` gồm
`kind` (`iso` | `arrow` | `leader` | `flag` | `bubble` | `generic` | `nested`), `label` (`fine_name`, `other_iso`,
`other_isa`, `other_generic`, `flow_direction` hoặc `None`), `template` (đã `oriented`, hoặc `None`),
`axis` + `forward` (mũi tên, cờ) và `flanges` (đĩa vẽ giữa hai mặt bích). Mỗi nút **một** lựa chọn, không trả danh sách xếp hạng (xem V5).
`configs/render_iso.yaml`, khóa `render_iso:` gồm `scope`, `p_replace`, `size_class`, đường dẫn
template và taxonomy, `flanges` (`p`, `classes`), và `rules`: danh sách phẳng, luật khớp đầu tiên
thắng (lọc theo `label`, `size`, `ports`, `next_to` / `not_next_to` (nhãn các nút mà ống dẫn tới:
có ít nhất một / không có cái nào); `groups` + `configs` + `classes` thu hẹp template; `replace`
tung `p_replace` trước); luật không
tìm được template thì rơi xuống luật sau; hết luật → `other_generic` + log. Bảng
`size_class × port_config` đã thêm vào `scripts/check_pid2graph.py`
(`size_class_by_port_config.csv`).

Chạy thử trên 12 tờ (2026-10-31): van 239 `iso` / 18 `generic` (đúng 18 van 0/1 cổng dự kiến);
`general` 205 `iso` / 28 `generic`; `instrumentation` 89 thay bằng G5 / 261 giữ bong bóng; `tank`
30 `iso` / 5 `generic` (bồn 0 cổng); `pump` 16/16 `iso`; arrow 449, cờ 96, lồng 3.

**`size_class` của đỉnh PID2Graph (chốt 2026-10-29)** — chỉ áp cho nhãn `general` (`valve`, `pump`,
`tank` đã tự xác định nhóm; kích thước của chúng chỉ quyết định tỉ lệ vẽ ở V5). Đo bằng px ảnh gốc,
**trước** khi phóng ×S: đỉnh `2_straight` lấy **cạnh vuông góc với ống** (cao nếu trục H, rộng nếu trục
V — phụ kiện trên ống chỉ cần mỏng theo chiều cắt ngang, nên đỉnh dài mảnh 14×75 px ở tờ 6 vẫn là
`small`); các cấu hình khác lấy cạnh dài.

```yaml
size_class:          # configs/render_iso.yaml — chỉ cho label general
  small_max: 25      # px ảnh gốc
  large_min: 120
```

| size_class | Ngưỡng | `general` (12 ảnh) | Ánh xạ |
|---|---|---|---|
| `small` | ≤ 25 px | 199 (85%) — `2_straight` 138, `1` 55, khác 6 | `2_straight` → G3; `1` → 511 cặp mặt bích ở đầu ống (từ 2026-11-05; trước đó G5); `3` → G3 (C0093); còn lại → `other_generic` |
| `medium` | 25–120 px | 21 (9%) — `2_straight` 10, `0` 5, `1` 4, `3` 2 | `other_generic` |
| `large` | ≥ 120 px | 14 (6%) — `multi` 7, còn lại rải rác | G4 (bồn/HX) theo số cổng |

Căn cứ: cả hai ngưỡng nằm trong **khoảng trống thật** của dữ liệu — cụm dày 10–24 px rồi nhảy lên
28 px (không đỉnh nào trong 24–28), và khoảng trống lớn nhất của cả dãy là 99 → 162 px — nên dời
ngưỡng vài px không đổi lớp đỉnh nào. Không chuẩn hóa theo tỉ lệ từng tờ: trung vị arrow lệch giữa các
tờ (11 px tờ 4 → 17 px tờ 8, ~1.5×) nhưng trung vị `general` nhỏ mỗi tờ vẫn 12–22 px, dưới ngưỡng.
`medium` không ép vào nhóm nào vì lẫn lộn: hộp 0 cổng 42–61 px ngoài đường ống, đỉnh 1 cổng 31–52 px,
đỉnh 2 cổng vuông 69×69 px có hai cổng lệch nhau 22–27 px (giống thiết bị quay), một đỉnh 3 cổng 99 px —
ép vào G4 thì bộ lọc có thể thành bơm. Ở `scope: html`, `large` hầu hết ra `other_iso` (chỉ 2511 trong
vocab). Số liệu đo bằng `utils.pid2graph` (`symbol_ports` + `port_config`) trên 12 ảnh; khi code V3,
thêm bảng `size_class × port_config` vào `scripts/check_pid2graph.py` để sinh lại được.

Bảng phân nhóm (bản 2 — ✅ duyệt 2026-10-29). Giữ cách chia 5 nhóm của đề xuất
taxonomy (file HTML user gửi) nhưng **sửa mã/tên**: mọi mã đã đối chiếu với 295 ký hiệu BK-2020, cột
"Cổng" đo từ stub trong SVG — kết quả cuối ở `data/raw/template_iso_svg/templates.csv` (62/62 khớp;
`port_audit.csv` + `port_audit_anomalies.png` chỉ là bản dò ban đầu). ✔ = có trong `vocab.csv` (Stage 2 đang dùng).

**Nhóm 1 — Van thẳng hàng (2 cổng LR)** → đỉnh `valve`, `2_straight`

| Reg | Tên (BK-2020) | Vocab | Ghi chú |
|---|---|---|---|
| 2101 | Valve (general) | ✔ | |
| X8068 | Valve, globe type | ✔ | |
| X8071 | Valve, ball type | ✔ | |
| X8074 | Valve, gate type | ✔ | |
| X8075 | Valve, butterfly type (Form 1, Form 2) | ✔ | Hai dạng vẽ chung **một mã** — coi là 2 biến thể của 1 lớp |
| X8076 | Valve, needle type | ✔ | |
| X8087 | Valve, control type, continuously operated | ✔ | 2 cổng LR (stub trái bị mũi tên chéo trùm — bắt được nhờ luật ≥ 25% nằm ngoài) |
| X8077 | Check valve (general) | ✔ | directional |
| X2113 | Check valve, globe type | ✔ | directional |
| X8078 | Swing check valve (Form 1) | ✔ | directional. HTML ghi nhầm là "ball check valve" — **BK-2020 không có van 1 chiều bi** |
| X8165 | Swing check valve (Form 2) | | directional |
| X2124 | Safety valve, spring loaded, globe type | ✔ | Đo được 2 cổng LR (HTML xếp nhóm 2) |

**Nhóm 2 — Van góc và 3 ngả** → đỉnh `valve`, `2_angle` / `3`

| Reg | Tên | Vocab | Cổng |
|---|---|---|---|
| 2102 | Valve, angle type (general) | ✔ | 2_angle (B+L) |
| X8069 | Valve, angle globe type | | 2_angle |
| X8072 | Valve, angle ball type | | 2_angle |
| X2125 | Safety valve, spring loaded, globe angle type | | 2_angle (B+R) |
| 2103 | Valve, three way type (general) | ✔ | 3 (B+L+R) |
| X8070 | Valve, three way globe type | | 3 |
| X8073 | Valve, three way ball type | | 3 |

**Nhóm 3 — Phụ kiện đường ống (2 cổng thẳng)** → đỉnh `general` nhỏ, `2_straight`

| Reg | Tên | Vocab | Ghi chú |
|---|---|---|---|
| 516 | Reducer | ✔ | HTML ghi nhầm 511 |
| 511 | Flanged connection | ✔ | Không stub — ống đi xuyên |
| C0096 | Orifice plate | | Không stub |
| C0097 / C0098 | Blind disc / Open disc | ✔ | Không stub |
| 2044 / 2045 | Interchangeable disc (blind / open) | ✔ | Không stub. **Vẽ dạng bích mù số 8** (tròn đặc + tròn rỗng) — đây chính là "spectacle blind" |
| 2034 | Viewing glass | ✔ | |
| 533 | Compensator | | |
| X8090 | Strainer | | |
| X8150 | Cone type strainer | | |
| X411 | Hose | ✔ | |
| X8174 | Piping, heated or cooled and insulated | ✔ | Ký hiệu dạng đoạn ống |
| C0093 | Mixing nozzle, injector | | **3 cổng** (L+R+B) — stub vẽ trùng, gộp tự động trong 1 mm |
| — | Spectacle blind (HTML: X8088) | | Mã sai — X8088 là van thở; bích mù số 8 chính là **2044/2045** ở trên |

**Nhóm 4 — Thiết bị lớn** → đỉnh `pump`, `tank`, `general` lớn

| Reg | Tên | Vocab | Cổng |
|---|---|---|---|
| 2301, 2322, 8091, X8095 | Pump: general / centrifugal / gear / diaphragm | ✔ | 2 (LR trên bản vẽ; khi đặt dùng đoạn gấp L) — directional. Mã đúng là **8091**, không phải X8091 |
| 2302, X8179 | Compressor: vacuum pump / centrifugal | ✔ | 2 |
| 2511, 2514 | HX straight tubes / coil | ✔ | 4 (2 trên, 2 dưới) |
| X8079 | Heat exchanger (general), condenser | ✔ | 4 (BLRT) |
| 2515 | Heat exchanger of double-pipe type | | 4 |
| 301, 2062 | Tank, vessel / vessel with dished ends | ✔ / — | multi — cổng tự do theo điểm chạm viền |
| 2063, X8100 | Spherical vessel / Column | ✔ | multi |
| C0082 | Electric motor | ✔ | Không đặt trên đường ống — chỉ gắn cạnh bơm/máy nén |
| 2672, C2024, C2027 | Agitators | ✔ | Không đặt trên đường ống — chỉ gắn vào bồn (1 cổng T là trục) |

**Nhóm 5 — An toàn, xả, đầu ống** → đỉnh 1 cổng (đầu ống) hoặc `2_straight`

| Reg | Tên | Vocab | Ghi chú |
|---|---|---|---|
| X8080 | Rupture disc | | Chỉ có 1 loại. HTML ghi X8081/X8082 = thiết bị tách / máy ly tâm |
| X8088 | Breather valve | ✔ | HTML ghi X8083 = lò sấy |
| 2036 | Flame arrestor | | HTML ghi X8084 = máy nghiền |
| C0087 | Flame arrestor, explosion-proof | | HTML ghi X8085 = máy đập |
| C0088 | Flame arrestor, detonation-proof | | HTML ghi X8086 = máy xay |
| C0091 | Flame arrestor, fire-resistant | | |
| 2181 | Self-operating release valve (steam trap) | | Chỉ có 1 loại. HTML ghi X2134/X8151 (không tồn tại), X8152 (= mặt bích đơn) |
| 2040 | Funnel | | **1 cổng thật** (thân là cổng, phía dưới). HTML ghi X8089 = kính quan sát có đèn |
| 2039 | Vent (outlet to atmosphere) | | **1 cổng thật** |
| 2033 | Silencer | ✔ | Vẽ 2 cổng LR |

Đỉnh 1 cổng (đầu ống): ứng viên = 2040, 2039 (1 cổng thật) **và** các thiết bị xả nhóm 5 vốn
2 cổng (X8080, X8088, 2036, C0087, C0088, C0091, 2033) đặt ở cuối ống, cổng còn lại để hở ra khí
quyển — đúng cách dùng thật của chúng. Bộ "sinh ba" có thật duy nhất: bộ ngăn lửa 2036/C0087/C0088.
**Từ 2026-11-05**: `general` nhỏ 1 cổng không còn đi G5 (trong OPEN100 chúng là cặp mặt bích ở đầu ống,
xem V5b); thiết bị G5 chỉ còn đến từ bong bóng 1 cổng được thay (`replace`).

**Lớp không có ký hiệu ISO**

| PID2Graph | Xử lý |
|---|---|
| arrow | 241 (template overlay, chỉ đầu mũi tên), xoay theo trục và chiều |
| instrumentation | Tỉ lệ `p_replace` (config, mặc định 0.3): đỉnh 1 cổng → thiết bị nhóm 5. Còn lại giữ bong bóng ISA, nhãn **`other_isa`** (cần cho đánh giá "Other"; dùng để train Stage 1 thì có thể bỏ nhãn = mẫu âm khó). Bong bóng 2 cổng (72) và 0 cổng (36) luôn giữ bong bóng |
| inlet/outlet | Mũi tên cờ đơn giản, không gán nhãn |
| không khớp luật | Hộp/tròn trơn, nhãn `other_generic` |
| ký hiệu lồng trong ký hiệu khác | Tâm nằm trong bbox của một ký hiệu khác và chỉ nối qua run `internal` (đo lại 2026-10-31: 2 arrow trong dàn ngưng ACC tờ 1, 1 `general` trong bồn tờ 11) → nội thất thiết bị: **không vẽ, không gán nhãn**, cùng với run `internal` của nó |

**Ký hiệu ngoài `vocab.csv` (cột Vocab trống) — ✅ chọn (a), 2026-10-29**: giữ nguyên `vocab.csv`
(42 dòng = 40 ký hiệu + `405 pipeline` + `241 flow_direction`), ký hiệu ngoài vocab gán nhãn
**`other_iso`** trong COCO của dataset mới → mẫu "Other" khó (cùng chuẩn, khác lớp). Không làm (b)
(mở rộng vocab → phải sửa B5, chọn lại H). Đổi sang (b) sau này chỉ cần thay `other_iso` bằng
`fine_name` — taxonomy đã có sẵn.

**Phạm vi vẽ — cột `scope` trong taxonomy (2026-10-29)**: render chỉ bốc template `scope = html`
(config `render_iso.scope: html`; `all` = cả 62). `html` = các ký hiệu của đề xuất 5 nhóm (file HTML),
đã sửa mã; cột `html_code` ghi mã HTML gốc để đối chiếu.

| | Số template | Ghi chú |
|---|---|---|
| `html` | 45 (G1 13, G2 7, G3 10, G4 7, G5 8) — 26 trong vocab, 19 → `other_iso`; **46 từ 2026-11-01**: thêm 511 `flanged_connection` (G3, trong vocab) cho vòi bình và mặt bích quanh đĩa; **47 từ 2026-11-02**: thêm 241 `flow_direction` (G0, overlay) cho mũi tên | Mã HTML sai đã thay: 511→**516** (reducer), X8088 spectacle→**2044 + 2045** (hai trạng thái của bích số 8), X8091→**8091**, X8081→**X8080**, X8083→**X8088**, X8084/85/86→**2036/C0087/C0088**, X8089→**2040**, X2134→**2181**. X8082 (đĩa nổ chân không), X8151/X8152 (bẫy hơi nhiệt/phao): không có trong BK-2020 → bỏ |
| `extra` | 17 → 16 | ~~511~~ (sang `html` 2026-11-01), C0098, X411, X8174, X8095, X8179, 2514, X8079, 301, 2063, X8100, C0082, 2672, C2024, C2027, C0091, 2039 — vẫn xuất SVG, không vẽ khi `scope: html` |

Hệ quả cần biết khi `scope: html`: (1) bồn/vessel PID2Graph chỉ có ứng viên 2062 (ngoài vocab) → mọi
bồn thành `other_iso`, Stage 2 không được thử trên lớp tank/column; (2) đỉnh 1 cổng thật chỉ còn
2040 (phễu) — các thiết bị nhóm 5 khác đặt ở cuối ống như đã mô tả; (3) motor/cánh khuấy (attach)
không vẽ. Muốn thử Stage 2 trên các lớp vocab bị loại thì chạy thêm một lượt `scope: all`.

**Trọng số bốc ký hiệu (2026-10-29) — ⏸ HOÃN (2026-10-31), hiện bốc đều.** Lý do hoãn: với luật
ở trên, các tập ứng viên gần như tách rời (valve `2_straight` → G1; `2_angle`/`3` → G2; general nhỏ
`2_straight` → **chỉ G3**; mọi nút 1 cổng → cùng một tập G5; pump/tank/general lớn → G4). Trong một
tập dùng chung, bốc đều đã cân bằng; phần lệch giữa các tập (G1 217 nút / 13 lớp so với G2 19 nút /
4 lớp) là do dữ liệu và trọng số không sửa được. Giữ trọng số thì phải truyền `counts` qua V3, V5
commit, V6 trừ khi bỏ biến thể — không đáng. **Bật lại** chỉ khi histogram ở V8 lệch quá 2× trong
một tập ứng viên. Công thức để dành:

```
P(c | nút) = (n_c + 1)^(-β) / Σ_{c' ∈ C(nút)} (n_c' + 1)^(-β)
```

- `C(nút)` = các template ứng viên của nút (theo khóa nhóm + khớp cổng); `c` tính theo `fine_name`.
- `n_c` = số mẫu lớp `c` đã sinh trên **toàn tập**, cập nhật ngay sau mỗi lần đặt; V6 bỏ một biến
  thể thì trừ lại.
- `β` = `render_iso.balance_beta` (khi bật, mặc định 1). `β = 0` = bốc đều; `β` lớn = ép cân bằng
  mạnh hơn.
- **Khi nào có ích**: chỉ khi một lớp nằm trong nhiều tập ứng viên có kích thước khác nhau (vd. nếu
  sau này cho G5 làm ứng viên cả cho `general` nhỏ `2_straight`).
- **Khác SynthPID**: họ dùng trọng số cố định cho từng loại (vd. van gate 0.45 / control 0.25 /
  check 0.2 / relief 0.1) và chỉ chia hạn mức theo độ phức tạp tờ (simple/medium/dense 25/50/25).
  Ta cân bằng theo lớp, vì đánh giá là theo lớp. Ý tưởng gần với repeat factor sampling của LVIS
  (Gupta et al., CVPR 2019).

> **Hỏi: dataset ISO đã cân bằng thì có còn cần few-shot cho phân loại mịn không?** — **Vẫn cần.**
> Few-shot tồn tại không phải vì mất cân bằng lớp, và cân bằng không thay được nó:
> 1. **Không thêm hình dạng mới.** Mọi mẫu của một lớp đều vẽ lại từ **một** template BK-2020
>    (X8075 có 2 hình). Train có giám sát trên tập này tương đương one-shot cộng augmentation, việc
>    B1 `SubpartWarp` đã làm. Cái thiếu thật sự là ảnh scan thật của ký hiệu ISO.
> 2. **Luận điểm của luận văn là lớp chưa thấy và mở rộng không cần train lại.** Theo P1, encoder
>    không được thấy ISO tổng hợp (§0). Train trên tập có đủ mọi lớp thì tập H mất nghĩa, và mỗi lần
>    thêm ký hiệu lại phải sinh dữ liệu rồi train lại.
>
> Vì vậy trọng số (nếu bật lại) chỉ phục vụ **tập test**, để mỗi lớp có đủ mẫu cho macro-F1 và ma trận nhầm lẫn
> từng lớp, và việc train **Stage 1** nếu V0 chọn vai trò đó. Nó **không** phục vụ việc train
> Stage 2. *Tùy chọn (chốt ở V0)*: train một bộ phân loại có giám sát chỉ trên các lớp **đã thấy**
> làm baseline/cận trên. Nếu one-shot đạt gần mức đó thì luận văn có thêm bằng chứng mạnh.

- Kiểm chỗ trống **chuyển sang V5** (V3 chưa có hệ số phóng mm→px và vị trí sau V4): ký hiệu đè
  ký hiệu khác **mà bản gốc không đè** → thu nhỏ tối đa 20%; vẫn đè → `other_generic`. **Không** thử ký hiệu khác trong nhóm
  (sẽ buộc V3 trả danh sách xếp hạng và V5 gọi lại V3); chỉ thêm nếu cổng G1 cho thấy đè là chuyện
  thường gặp.

**Kiểm**: test luật (mọi khóa đều trả về ít nhất 1 template hoặc fallback; 3 nguyên tắc; hướng trả
về có cạnh cổng khớp nút); histogram **theo `fine_name`** qua nhiều seed lệch ≤ 2× giữa lớp nhiều nhất và ít
nhất trong G1 (bốc đều theo lớp phải tự đạt).

## V4 — Biến đổi bố cục giữ topology 🤖 (~0.5 ngày) — 🟢 XONG 2026-10-31 (17 test)

**Code**: `src/utils/iso_layout.py` (hàm thuần, như `pid2graph.py`), test ở `tests/test_iso_layout.py`.
`transform_layout(sheet, runs, rng, cfg) -> (Sheet mới, list[PipeRun] mới)`. Config thêm vào
`configs/render_iso.yaml` (không đụng `configs/config.yaml`):

```yaml
render_iso:
  layout:
    scale: 3                 # S
    column_tol: 3            # px ảnh gốc
    gap_scale: [0.85, 1.15]  # hệ số nhân thêm vào S cho khoảng giữa hai cột liền kề
```

**Thứ tự pipeline: V2 → V3 → V4 → V5.** V3 chạy trên px gốc, vì ngưỡng `size_class` đo trước khi
phóng ×S. Kết quả V3 vẫn dùng được sau V4: ánh xạ giữ thứ tự nên cổng không đổi cạnh, và
`symbol_ports` lấy cổng từ đầu/cuối run cùng `a_side`/`b_side` đã lưu (không đo lại theo dung sai px),
nên V4 không cần ánh xạ riêng cổng.

**Ánh xạ trục đơn điệu** (làm riêng cho x và y):
- Gom mọi tọa độ (mép bbox, tâm connector/crossing, điểm polyline, mép `background`, 0 và W) thành
  **cột**: duyệt từ nhỏ đến lớn, mở cột mới khi vượt `column_tol` tính từ **mốc đầu cột** (không nối
  chuỗi, để một dãy giá trị cách nhau 1–2 px không thành một cột dài). Mọi giá trị trong một cột ánh
  xạ về cùng một vị trí → không biến dạng, và các gấp L lệch 1–3 px tự suy biến.
- Khoảng giữa hai cột liền kề = khoảng gốc × S × hệ số bốc đều trong `gap_scale`.
- **Ký hiệu giữ cứng**: khoảng giữa hai cột mà một bbox ký hiệu bắc qua (`xmin` ở cột này, `xmax` ở cột
  kia) luôn nhân đúng S, không jitter → valve 20 px thành ~60 px, V5 không phải co template theo bbox
  méo (sai số ±`column_tol`·S do mép bbox dồn về mốc cột). Khoảng giữa các ký hiệu và đoạn ống tự do
  thì co giãn. **Không tính `background`** là ký hiệu: cả 12 tờ có nút `background` phủ gần hết tờ;
  tính nó thì mọi khoảng đều cứng và không còn jitter.
- Khác SynthPID: họ ánh xạ **tâm** ký hiệu (khoảng × 0.84–1.2), dời bbox nguyên khối rồi co giãn kích
  thước theo nhãn (với OPEN100 còn dùng hàm riêng `_normalize_open100_scene`). Ta ánh xạ mọi tọa độ
  qua cùng một trục, nên đầu ống vẫn dính mép ký hiệu.

**Đầu ra**: đỉnh ký hiệu — bbox ánh xạ qua trục; `connector`/`crossing` — tâm ánh xạ, kích thước ×S
(8×8 → 24×24); `background` — các mép ánh xạ; canvas = ánh xạ của (W, H). Giữ nguyên id, nhãn, danh
sách cạnh. Polyline của run ánh xạ từng điểm rồi chạy lại `_simplify`; run `nozzle` (hai điểm trùng,
dài 0) không `_simplify` để khỏi mất điểm. Không xóa/chèn đỉnh, không đổi nhãn.

**Kiểm** (test):
- Mọi đoạn ống vẫn song song trục.
- Đồ thị **trùng khớp** bản gốc: cùng tập id đỉnh, cùng nhãn, cùng multiset cạnh (V4 giữ nguyên id nên
  so trực tiếp bằng dict — chặt hơn đẳng cấu, không cần networkx).
- **Cặp ký hiệu đè nhau: không có cặp mới; một cặp cũ chỉ được mất nếu phần đè ban đầu ≤
  `column_tol`.** Không dùng "không có hai ký hiệu đè nhau": bản gốc đã có **127 cặp** đè nhau trên 12
  tờ (instrumentation–valve 40 = bộ truyền động, instrumentation–instrumentation 34, arrow–general 18,
  general–tank 18, …), trong đó 67 cặp chỉ đè ≤ 3 px — gom cột có thể đưa hai mép đó về cùng một vị
  trí nên cặp biến mất. Test kiểm cả hai chiều và đòi `lost > 0`, để nhánh "cặp mất phải nông"
  chắc chắn được chạy.

**Kết quả trên 12 tờ × 3 seed (review 2026-10-31)**: 0 đoạn chéo; 0 cặp đè mới; ~51 cặp mất mỗi seed,
tất cả đè ≤ `column_tol`; 0 cổng đổi cạnh (`symbol_ports` chạy lại trên tờ mới); 0 nút có chiều
≤ `column_tol`. Hai điều V5 phải biết:
- **Kích thước ký hiệu méo**: tỉ lệ (mới / S × cũ) trên 8634 chiều bbox có trung vị 1.00, p1 0.76,
  p99 1.26, tệ nhất 0.53 (`arrow120` tờ 6, cạnh 6.2 px); ~5% lệch quá 20%. Nguyên nhân: mép bbox dồn
  về mốc đầu cột, lệch tới 3 px — với ký hiệu nhỏ là tới 50%. Không sửa V4 (gom cột là cái giá để dồn
  các gấp L lệch 1–3 px); V5 lấy **kích thước** từ bbox gốc × S và **vị trí** từ cổng đã ánh xạ.
- **Run dài 0**: ~262 run mỗi seed co về độ dài 0, tất cả dài 0–4 px ở bản gốc (arrow–crossing 166,
  van–bộ truyền động 41, general–tank 21; không run `internal` nào). Đây là các vật vốn đã chạm nhau.
  `AxisMap` chặn chỉ số ≥ 0, nên giá trị dưới mốc đầu tiên không bị đẩy về cuối trục.

## V5 — Vẽ cả tờ 🤖 (~1.5 ngày) ⛔ G1 — thiết kế chốt 2026-10-31 — 🟢 G1 ĐẠT 2026-11-02

**Code**: `src/utils/page_svg.py`:
`render_page(sheet, runs, assignments, cfg) -> Page`, với `Page` gồm `svg` (một chuỗi cho cả tờ),
`annotations` (list dict: id nút, `label`, `kind`, `fine_name`, bbox **mực**) và log chống đè;
`rasterize(svg, dpi=72)` → PNG xám, 1 đơn vị SVG = 1 px, nên ảnh bằng canvas sau V4 (~6000–10000 px
mỗi chiều khi S = 3 — nhớ khi tính chi phí V7). Ghi `annotations.coco.json` + GraphML ra thư mục là
việc của **V8**, không ở V5. `scripts/check_g1.py` đo 3 tiêu chí tự động của G1. Config thêm vào
`configs/render_iso.yaml`:

```yaml
render_iso:
  page:
    stroke_px: 3            # giữa [2, 4]·S/3 của V6 khi S = 3
    junction_dot: false     # ngã ba: chấm tròn tùy chọn; bắt chéo: không bao giờ chấm
    shrink_steps: [0.9, 0.8]
```

**Phạm vi lượt này (G1)**: mọi template có **cổng cố định** (G1–G5, gồm bơm `2` và HX `4`) dùng chung
một thuật toán: đặt theo cổng, rồi nối ống. **Bình `multi` tạm vẽ thành hộp `other_generic`** (đúng ra
là `other_iso`): 2062 ngoài vocab nên Stage 2 không đánh giá lớp này, tờ 3 chỉ có 3 bình, còn kéo ống
tới viền bình phải dò mực theo từng lần đặt. **Làm ở V5b** (sau G1, trước V6 — đổi 2026-11-02; trước đó ghi "trước V8"): vẽ 2062 vừa bbox gốc ×S,
kéo mỗi đầu ống dọc trục cổng tới viền thật (dò mực), tia trượt ra ngoài thì kẹp vào mép bình rồi nối
đoạn gấp. **Agitator** (2672, C2024, C2027 — `attach`, hiện không vẽ ở đâu) làm cùng lúc: vẽ
**trong** bình, trục đứng xuyên đỉnh bình, cánh khuấy bên trong, có nhãn COCO riêng. PID2Graph không
có nút agitator, nên V8 phải thêm nó (vào COCO, và có thể vào GraphML như một nút gắn với bình).

**Đặt template** (thứ tự: đặt → chống đè → nối ống, vì thu nhỏ làm cổng dịch chỗ):
- Mỗi ký hiệu là `Template.body(stroke=W/k)` của template đã `oriented`, bọc trong
  `<g transform="placement · matrix">` → nét ký hiệu = nét ống = `stroke_px`.
- **Kích thước** `k` (px/mm) = S × min(bbox_w / ink_w, bbox_h / ink_h), với bbox **gốc** (trước V4) và
  hộp mực thật của template **theo hướng đã xoay** (90°/270° đổi chỗ w/h; hộp mực tính một lần ở hướng
  gốc, cache, rồi biến đổi theo ma trận — không dùng viewBox có đệm 0.5 mm). Template đồng dạng, lọt vừa
  bbox gốc ×S. Van `2_straight`: trung vị bbox dọc trục 22 px, 2101 có khoảng cách cổng 10.2 mm → k ≈
  6.5 px/mm. Bbox sau V4 méo tới p1 0.76 / p99 1.26 (xem V4) nên không dùng để tính kích thước.
- **Cỡ đĩa và cặp mặt bích** (orifice plate, blind/open disc, interchangeable disc —
  `flanges.classes` — và 511; chốt 2026-11-02): vẽ theo **tỉ lệ mm của van trên cùng tờ**
  (`_sheet_scale`: trung vị `k` của các van G1, mỗi van vừa bbox của nó). Lý do:
  - BK-2020 vẽ mọi ký hiệu ở một tỉ lệ (thân van, vạch mặt bích, đầu đĩa đều 5 mm), còn bbox
    PID2Graph không nói gì về chúng: node của đĩa là một cặp mặt bích (khoảng 13×8 px).
  - Hai lần thử trước đều lệch. Lần 1: thu cả cây kẹo mút lọt bbox, k khoảng 2.6, đĩa nhỏ khoảng 3
    lần so với van. Lần 2: đầu đĩa = cạnh node ngang ống, k khoảng 8.5; đĩa cao 154 px, khoảng 4 lần
    thân van (38 px).
  - Soát OPEN100: ở 6 tờ có bbox van chỉ bao thân van, cặp mặt bích **cao xấp xỉ thân van** (tỉ lệ
    0.92–1.10).
  - Kết quả seed 0 trên 12 tờ (đo ngang ống): van 38 px, 511 34 px (tỉ lệ 0.9), đĩa 113 px (đầu khoảng
    30 px ngang thân van; cán chìa ra theo đúng tỉ lệ ISO: cán khoảng 11 mm, đầu 5 mm). Một cặp mặt
    bích có cùng cỡ dù có đĩa hay không.
  - Tờ không có van G1: đĩa lấy đầu đĩa = cạnh node ngang ống (`_disc_scale`), 511 lấy theo bbox. Đĩa
    còn được thu nhỏ khi chật, xuống tới cỡ vừa cả bbox (xem "Chống đè").
- **Thiết bị G5 thay bong bóng** (luật `replace`, 2026-11-01): **không** lấy cỡ theo bbox bong bóng
  (trung vị 48 px gốc), vì trước đây chúng được vẽ to gấp ~5 lần cùng thiết bị trên node `general`
  (trung vị 146 px so với 28 px). Thay vào đó lấy theo một box vuông `page.replaced_size` = 14 px gốc,
  đúng trung vị cạnh dài của G5 trên `general`, đặt ở tâm bbox gốc. Log `resized`; seed 0: 38.
- **Ghép cổng** template ↔ nút theo cạnh; nhiều cổng cùng cạnh (HX 2511: 2 T, 2 B) xếp theo vị trí dọc
  cạnh; cổng không trùng cạnh (bơm) ghép theo khoảng cách gần nhất. Cổng thừa của template (G5 ở cuối
  ống) để hở.
- **Vị trí** = trung bình độ lệch giữa cổng template và đầu ống (sau V4) của các cổng đã ghép.

**Nối ống** (ống dừng đúng tại tọa độ cổng — `(x_mm, y_mm)` của `ports.csv` qua cùng transform):
- Độ lệch còn lại **dọc trục ống** → kéo dài hoặc cắt đoạn ống cuối (thẳng hàng, không gấp). Đây là
  phần lệch thường gặp nhất, vì template đúng ×S còn bbox sau V4 thì méo. Không được cắt lùi qua đỉnh
  trước của polyline.
- Độ lệch **ngang trục** → chèn đoạn gấp Z/L ngắn, không bao giờ nối chéo. Bơm (cổng hút ở tâm một bên,
  cổng đẩy tiếp tuyến phía trên) và thiết bị lớn đi theo luật này.
- Run dài 0 sau V4 (~262 trên 12 tờ mỗi seed, vật vốn chạm nhau): không vẽ ống, hai cổng trùng nhau.
- **Vòi bình** (2026-11-01): run dài 0 (hoặc thoái hoá) giữa một ký hiệu ISO và một hình vẽ dạng
  hộp (bồn đang vẽ tạm bằng hộp, `general` không có luật) là vòi bình: ở bản gốc, mặt bích hay vòi
  bị bịt áp sát thành bình. Ký hiệu ISO hẹp hơn bbox, nên trước đây nó cách bồn 2–9 px (phụ kiện 2
  cổng), hoặc treo ngay trên viền bồn và đè lên viền 1 nét (thiết bị G5 1 cổng: 14 ca, trông như lệch
  khỏi ống). Giờ `_nozzle_links` tìm các liên kết này. Thiết bị 1 cổng bị đẩy ra xa thành bình
  `page.nozzle_stub` × cỡ mực dọc trục (0.5). Sau đó `_nozzle_stub` vẽ ống từ cạnh hộp tới cổng
  (phụ kiện 2 cổng giữ nguyên chỗ, chỉ bắc ống qua khe). Log `nozzles`; seed 0 trên 12 tờ: 24.

**Chống đè** (luật chuyển từ V3, 2026-10-31): xét mọi cặp ký hiệu **không** đè nhau ở bbox PID2Graph
gốc (127 cặp gốc — vd. 40 cặp van–bộ truyền động — là bố cục thật, áp luật cho chúng sẽ biến van thành
hộp trơn). Cặp nào đè nhau theo **hộp mực** của ký hiệu đã đặt: thu nhỏ ký hiệu ISO có diện tích mực lớn
hơn theo `shrink_steps` (×0.9 rồi ×0.8, tối đa 20%) quanh tâm của nó; vẫn đè → thay bằng `other_generic`
(hộp theo bbox sau V4). Cặp mà cả hai đều không phải ISO thì bỏ qua. Ghi log số ca mỗi tờ để biết có
cần thêm bước thử ký hiệu khác hay không.
**Sửa 2026-11-01** (đĩa to lên, xem "Cỡ đĩa và cặp mặt bích"): (1) **xoay trước khi thu nhỏ**: thử các hướng khác của
cùng template có cùng cạnh cổng (với đĩa: cán sang phía bên kia ống), nhận hướng đầu tiên không đè cặp
mới nào (log `turned`). (2) Thu nhỏ **quanh tâm các cổng** của template, không quanh tâm mực, nên ống
vẫn thẳng (với đĩa: đầu cán đứng yên). (3) Đĩa được thu tiếp sau `shrink_steps`, mỗi lần 0.1, xuống
tới cỡ cũ (vừa cả bbox) trước khi bị thay bằng hộp, vì đĩa nhỏ vẫn hơn hộp. (4) Hình của đĩa
gồm cả **mặt bích** của nó (mặt bích thu nhỏ cùng đĩa). Seed 0 trên 12 tờ, sau khi đĩa theo tỉ lệ
của van: 0 bị thay bằng hộp, 3 bị thu nhỏ, 6 được xoay (lần đo khi đĩa còn to: 10 đĩa bị thay bằng
hộp).

**Hình cho các loại không phải template** — vẽ theo bbox **sau V4** của nút, vì đầu ống đã nằm trên
mép đó:
- `arrow` (`flow_direction`): ống đi từ mỗi cổng tới tâm bbox bằng các đoạn song song trục (ống không bị
  cắt; arrow `2_straight` là trường hợp riêng: ống xuyên qua), đầu mũi tên phủ lên tại tâm, dọc `axis`,
  chiều theo `forward`. **Vẽ bằng template 241** (2026-11-02; `_place_overlay`): cỡ khớp bbox gốc của
  chính mũi tên ×S, vì khác với đĩa, PID2Graph có box thật cho mũi tên; bbox COCO = hộp mực của đầu
  mũi tên, nhãn `flow_direction`, `template` 241. Không có template thì vẽ tam giác lấp đầy bbox. **Đỉnh
  đặt trên cổng phía đỉnh** (mép hộp mực, vì cổng dò trên mực nằm lùi vào trong mũi nhọn khoảng 0.2
  mm): nên với mũi tên sát chỗ gấp, đỉnh chạm đúng chỗ gấp như trong OPEN100.
- `leader` (`leader_arrow`, 2026-11-02; `_leader_shape`): một nét thẳng mảnh (nửa độ dày ống) từ đầu
  kia của đường chỉ dẫn tới đáy đầu mũi tên. Nét này chéo như trong OPEN100, không theo đường đã nắn
  chữ L của V2. Đầu mũi tên là template 241 xoay theo **góc bất kỳ**, căn giữa bbox của node (sát
  ống) và trỏ ra xa đầu kia của đường. Không vẽ ống cho cạnh này. Bbox COCO = hộp của đầu mũi tên
  đã xoay. Seed 0 trên 12 tờ: 72. **Đặt trên ống** (sửa
  2026-11-02): bbox PID2Graph của mũi tên không đối xứng quanh đường ống, nên trước đây 347/449 mũi
  tên lệch khỏi ống hơn 1 px (trung vị 2 px; tối đa khoảng 30 px sau V4), và ống phải gấp một khúc
  vào tâm. `_arrow_box` dời bbox **ngang trục** về đường ống của cổng nằm trên trục mũi tên (có hai
  đường thì lấy đường gần tâm bbox nhất); dọc trục giữ nguyên tâm bbox. Bbox annotation và bước chống
  đè dùng box đã dời. Kết quả: 0/449 lệch. 13 mũi tên `2_straight` có khúc gấp thật ở bản gốc (> 15
  px, V4 không nối thẳng): mũi tên nằm trên một đường, đường kia gấp vào.
- `flag`: mũi tên ngũ giác đơn giản (như hình nhà) theo `axis` và `forward`, ống dừng ở mép cờ.
- `bubble` (`other_isa`): hình tròn nét viền nội tiếp bbox. Đầu ống nằm ở chỗ connector chạm bbox, còn
  vòng tròn chỉ chạm bbox ở 4 điểm giữa cạnh → **kéo mỗi đầu ống dọc trục tới vòng tròn** (giao điểm
  tính bằng công thức), không thì tiêu chí G1 số 1 trượt trên mọi bubble (tờ 3 có 24).
- `generic` (`other_generic`): hình chữ nhật nét viền đúng bbox.
- `nested`: không vẽ.
- **Mặt bích quanh đĩa** (`Assignment.flanges`, 2026-11-01): `_place_flanges` vẽ **chính template
  511** (đã xoay ở V3), tâm giữa hai cổng của nó đặt trùng đầu cán (chỗ hai cổng L/R của đĩa). Như
  vậy mặt bích quanh đĩa trông giống hệt 511 đứng riêng. **Cỡ**: giống 511 đứng riêng, theo tỉ lệ mm
  của van trên tờ (xem "Cỡ đĩa và cặp mặt bích"); thu nhỏ cùng đĩa. Có mức sàn: mỗi vạch cách cán ≥ 2
  nét (`_FLANGE_PITCH_MM` 2.5 → k ≥ 4.8 px/mm khi nét 3 px); ở S = 3 mức sàn không bao giờ có tác
  dụng. `check_g1` tiêu chí 5 coi hình vẽ của một node là **hợp** các box của nó (đĩa ∪ mặt bích),
  vì ống dừng ở mặt bích, mà mặt bích cố ý tách khỏi cán. Mặt bích được đặt **trước** khi nối ống, và
  ống dừng ở **mặt ngoài vạch mặt bích** (cổng của 511), như khi 511 đứng riêng; giữa hai vạch
  chỉ còn cán đĩa, không có đoạn ống thừa (sửa 2026-11-01).
  **Annotation riêng**: `id = "<node>/flanges"`, `label = flanged_connection`, bbox = hộp mực của
  511 đã đặt. Mọi annotation có thêm trường `node` (nút PID2Graph mà nó đại diện; = `id`, trừ phần
  mặt bích). `check_g1` gom theo `node`: đĩa và mặt bích của nó không tính là cặp đè mới. Log
  `flanges`; seed 0 trên 12 tờ: 34 đĩa có mặt bích. V8: một nút đồ thị có 2 box COCO.

**Khác**:
- Ngã ba: chấm tròn tùy chọn (`junction_dot`); bắt chéo: không chấm.
- Chữ/tag: **chưa làm ở G1** (OPEN100 không có nhãn chữ); v2 sinh tag cạnh ký hiệu.
- Kiểm khi xem ảnh: 4 arrow trong bbox dàn ngưng ACC tờ 1 có ống ra ngoài nên vẫn được vẽ, trong khi
  ACC (`general` lớn) được vẽ bằng bình 2062 → mũi tên và ống có thể nằm đè trong thân bình.

**Tiêu chí G1** (tự động bằng `scripts/check_g1.py` + 🧑 xem ảnh), chạy trên **ảnh 3**, đo trên **mọi
ký hiệu ISO** của tờ (seed 0 trên tờ 3: 15 van G1, 8 G3, 16 G5, 2 bơm; 3 bình tạm là hộp) và mọi
bubble:
1. Mọi cổng có ink ở cả phía ống và phía ký hiệu trong bán kính 1 px (không hở).
2. Độ dày nét đo bằng distance transform (cách đã dùng 2026-10-26): ký hiệu và ống chênh ≤ 1 px.
3. Không có **cặp** bbox ký hiệu đè nhau **mới** so với bản gốc (bản gốc đã có 127 cặp đè nhau trên 12
   tờ — xem V4).
5. (thêm 2026-11-01) Mỗi cạnh đồ thị giữa hai ký hiệu được vẽ phải có mực nối, hoặc hai ký hiệu
   chạm nhau (`edges_joined`). Tờ 3 đạt 45/45. Lần đo đầu, 12 tờ seed 0 trượt 9 cặp, phần lớn là
   **liên kết dài 0 giữa phụ kiện ở vòi bình và bồn** (hở 2–9 px). Đã sửa bằng ống vòi (xem "Vòi bình"
   ở trên). `general73–general75` (tờ 4) hết sau khi đĩa theo tỉ lệ của van; arrow–general (tờ 10)
   hết sau khi mũi tên vẽ bằng 241 (2026-11-02). **Còn mở 3 cặp**: bong bóng–bong bóng (tờ 0, 7, 9).
   Sửa trước khi chốt G1.
4. 🧑 duyệt ảnh. **Không đạt → dừng nhánh này.** Nếu trượt chỉ vì một nhóm (bơm, G2) thì sửa nhóm đó
   trước khi kết luận no-go: G1 đánh giá hướng tiếp cận, không phải từng nhóm.

**Kết quả G1 — 🟢 ĐẠT (user duyệt ảnh 2026-11-02)**. Tờ 3, seed 0: cả 4 tiêu chí tự động (1, 2, 3, 5)
đạt; tiêu chí 4 (duyệt ảnh) do user chốt. Đo thêm trên cả 12 tờ OPEN100, seed 0 (`scripts/check_g1.py
--sheet i`; 476 test pass):

| Tiêu chí | 12 tờ | Ghi chú |
|---|---|---|
| 1. Ống và ký hiệu gặp nhau ≤ 1 px | 1357/1361 cổng → **1361/1361** (2026-11-02) | 4 cổng trượt (tờ 4, 6, 7) vì `_stems` bắt đầu ở tâm ký hiệu rỗng (511, đĩa nổ); đã sửa (xem "Trước V6", mục 1) |
| 2. Độ dày nét ký hiệu vs ống ≤ 1 px | 12/12 tờ | 3.00 / 3.00 px |
| 3. Không có cặp đè mới có ký hiệu ISO | 12/12 tờ | 490 ký hiệu ISO; 3 bị thu nhỏ, 6 được xoay, 0 bị thay bằng hộp |
| 5. Cạnh đồ thị có mực nối hoặc chạm | 658/660 | 2 cặp bong bóng–bong bóng hở (tờ 7, 9); ngoài ra 14 cặp bong bóng đè nhau 5–18 px mà các tiêu chí không bắt — xem "Trước V6", mục 2 |

Những gì đã sửa trong vòng G1 (2026-10-31 → 2026-11-02; chi tiết ở các mục trên): khúc gấp ở bơm
và đường thân bộ truyền động (`_stems`); lệch trục ống quanh fitting thẳng (V4 `straight_max`); liên kết
bong bóng–ISO bị bỏ vì box chồng nhau; luật vòi bình (không orifice/strainer/kính quan sát ở vòi bình);
cổng đĩa ở đuôi cán (V1 `@tail`); 511 vào scope, gỡ đoạn ống thừa (V1 `pipe_pieces`), mặt bích quanh đĩa
có nhãn riêng, ống dừng ở mặt bích; ống vòi bình (`_nozzle_links`); G5 thay bong bóng lấy cỡ fitting;
đĩa và 511 theo tỉ lệ của van trên tờ; xoay trước khi thu nhỏ, thu nhỏ quanh cổng; mũi tên đặt trên
ống, vẽ bằng template 241, chiều theo bản vẽ khi sát chỗ gấp; mũi tên chỉ dẫn (`leader_arrow`).

**Trước V6 (theo thứ tự)**:
1. ✅ **XONG 2026-11-02, kể cả phần 511 (xem ✅ dưới)** — sửa `_stems` (`src/utils/page_svg.py`): van G1/G2 giữ điểm bắt đầu ở tâm
   (thân bộ truyền động vào tâm van); ký hiệu khác bắt đầu ở viền mực gần bong bóng nhất
   (`_Placed.ink_nearest`: điểm mực gần tâm bong bóng nhất, nên đoạn nối không cắt mực nào khác của ký
   hiệu — không đi xuyên thân G5, không bắt đầu từ khe của 511). Đầu ống lùi vào trong nét viền nửa nét
   (`w_px/2`): đầu ống cắt phẳng, nếu dừng đúng tâm nét thì ở tờ 2 một cổng (silencer 2033) trượt
   tiêu chí 1 do lệch nửa pixel. Kết quả seed 0, 12 tờ: tiêu chí 1 **1361/1361**; tiêu chí 2, 3 không
   đổi; tiêu chí 5 còn 2 cặp bong bóng–bong bóng (tờ 7, 9 — mục 2). Test mới
   `test_a_stem_to_a_hollow_fitting_starts_on_its_ink_nearest_the_bubble` (fail khi tắt bản sửa);
   477 test pass.

   **✅ Mở lại cho 511 (chốt và code 2026-11-02).** Với 511, bản sửa trên khác bản gốc: đường bắt
   đầu ở một mặt bích nên bị xiên (tờ 6, 7). Ở OPEN100, đường tín hiệu luôn vuông góc với ống và đi vào
   **khe giữa hai mặt bích**: 1/4 mẫu dừng ở trục ống, 3/4 kéo nét mảnh xuyên hết khe (nét đó là tấm
   orifice). Trước bản sửa, đường từ tâm 511 giống bản gốc; nó trượt tiêu chí 1 chỉ vì khe không có mực,
   tức là tiêu chí bắt sai chứ hình không sai.
   Cả 11 fitting nhỏ `2_straight` có gắn bong bóng trong 12 tờ đều là **cặp mặt bích trống** với bong
   bóng FO/RO (orifice đo lưu lượng / tiết lưu). Seed 0 đang vẽ 8/11 sai: 4 đĩa (2 có mặt bích),
   1 reducer, 3 `other_iso`.
   - [x] **V3: có bong bóng thì luôn là 511.** Luật mới đặt trước luật vòi bình:
         `{label: general, size: small, ports: ["2_straight"], tagged: true, groups: [G3], classes: [flanged_connection]}`.
         Bộ lọc mới `tagged`: có run nối node với một nút `instrumentation`. Không dùng được `next_to`
         vì `symbol_ports` bỏ qua bong bóng.
         Hệ quả: fitting có bong bóng không bao giờ là đĩa (đứng riêng hay kẹp mặt bích) hay G3 khác,
         và ngược lại đĩa chỉ rơi vào fitting không có bong bóng. Chỉ 11 node đổi nhãn.
   - [x] **V5 `_stems`:** với 511, đường bắt đầu ở **tâm khe, trên trục ống**, đi vuông góc với ống
         ra phía bong bóng; chỉ khi tia vuông góc không trúng vòng tròn mới đi theo đường nối hai tâm.
         Đường **dừng ở trục ống**, không kéo xuyên khe: kéo xuyên sẽ trông như orifice_plate kẹp mặt
         bích (một lớp có thể có mặt bích), mà dừng ở trục ống vẫn có trong OPEN100.
   - [x] **`check_g1` tiêu chí 1:** chấp nhận đầu đường tín hiệu nằm trong khe 511 (trên trục ống,
         giữa hai mặt bích).
   - Giữ nguyên phần còn lại của bản sửa: van G1/G2 bắt đầu từ tâm; G5 và đĩa nổ bắt đầu từ viền mực
     gần nhất.
   - Kiểm: tiêu chí 1 vẫn 1361/1361; tiêu chí 2, 3, 5 không xấu đi; test mới cho luật `tagged` và cho
     đường vuông góc vào khe 511; ảnh trước/sau ở `docs/experiments/iso_render/stems_fix/`.
   - **Kết quả (seed 0, 12 tờ):** cả 11 fitting nhỏ có tag đều thành 511 (đo thêm seed 0–2: 33/33, không
     có đĩa hay mặt bích kèm). Tiêu chí 1 **1361/1361**; tiêu chí 2, 3 đạt như cũ; tiêu chí 5 vẫn 2 cặp
     bong bóng (tờ 7, 9 — mục 2). 485 test pass, ruff sạch.
     Code: luật `tagged` đặt trước luật vòi bình trong `configs/render_iso.yaml`, bộ lọc `tagged` trong
     `iso_mapping.assign_symbols`; `_gap_of` / `_gap_stem` trong `page_svg.py`; `page.ports` có thêm khóa
     `gap: {axis, half}` cho đầu đường nằm trong khe, `check_g1` đòi mực mặt bích ở **cả hai phía** trên
     trục ống thay vì đòi chạm mực. Khi tia vuông góc trượt vòng tròn thì đi theo đường nối hai tâm
     (xiên) từ cùng điểm khởi đầu.
   - **Vòng sửa 2 (2026-11-03, từ review 5 seed):** hai lỗi, đã sửa.
     (1) Tag của 511 bị `replace` thành thiết bị G5 (12/55 tag ở 5 seed): ống thành đầu lỏng, bám mực
     gần nhất và gấp khúc, và FO/RO là tên tấm orifice nên G5 ở đó sai nghĩa. Luật `replace` nay
     `not_next_to: [valve, general]`. Đổi lại 13 bong bóng gắn với node `general` khác không còn bị
     thay (G5 thay bong bóng giảm từ ~72 xuống ~42 trên 12 tờ; `test_real_sheets_counts_match_the_plan`
     tính theo số ứng viên). Kết quả: 0/55 tag bị thay.
     (2) Bong bóng lệch: đường thẳng từ tâm khe tới vòng tròn cắt qua mặt bích (3/3 seed có đường
     này ở tờ 2 `general26`). `_gap_stem` nay trả **hai đoạn** khi tia vuông góc trượt vòng tròn:
     vuông góc với ống ra khỏi khe, quá đầu hai vạch một nét (`w_px`), rồi thẳng tới vòng tròn.
     Kết quả: 0 đường cắt mực. Ca bong bóng ngang tâm khe vẫn một đoạn.
     `check_g1` thêm **tiêu chí 6**: đường tín hiệu không cắt mực ký hiệu ngoài 6 px ở mỗi đầu (đường
     bắt đầu từ tâm van bỏ ra). `Page.stems` mới (id, bubble, points, from_centre). Thử đặt lại đường
     thẳng cũ thì tiêu chí 6 trượt 1/6 ở tờ 2, nên kiểm tra có tác dụng. Seed 0, 12 tờ: tiêu chí 1
     đạt cả 12 tờ, tiêu chí 6 đạt cả 12 tờ; tiêu chí 5 vẫn 2 cặp bong bóng (mục 2). 487 test pass.
     Ghi chú: tờ 4 `general82`, bong bóng FO 1662 chạm 3 px vào hộp `valve89` (PSV bị gán `valve`, vẽ
     hộp generic): bản gốc vốn cách nhau 1 px, không có cạnh đồ thị giữa chúng, thuộc mục 2.
   - **Vòng sửa 3 (2026-11-04): bong bóng lệch → đường chéo từ chỗ ống gặp mặt bích.** Đường hai đoạn
     của vòng 2 có một khúc ngắn ra khỏi khe rồi mới bẻ chéo; OPEN100 không vẽ thế. Bản gốc ở đúng ca
     này (tờ 2, RO 1603) kéo đường chéo **bắt đầu ở giao điểm ống và mặt bích phía bong bóng**, không
     vào khe. "Ngang hẳn" thì không làm được: tia ngang không trúng vòng tròn, phải dời bong bóng (đổi
     V4) hoặc gấp chữ L, bản gốc không làm cả hai.
     `_gap_stem` nay: khi tia vuông góc trượt vòng tròn, điểm đầu là **cổng của vạch phía bong bóng**
     (tâm khe ± nửa khe theo trục ống, phía bong bóng), đi **một đoạn thẳng** tới vòng tròn. Đường rời
     vạch từ mặt ngoài nên không cắt vạch nào. Điểm đầu nằm trên mực (vạch + ống), nên `page.ports`
     **không** gắn `gap` cho ca này: tiêu chí 1 kiểm như cổng thường. Ca vuông góc vào khe giữ nguyên.
     Test `test_a_flange_pair_tag_off_to_the_side_starts_where_the_pipe_meets_the_mark` (thay test
     hai đoạn): điểm đầu ở cổng phía bong bóng, đúng 2 điểm, nhắm tâm vòng tròn, không cắt mực, có mực
     vạch và ống ở điểm đầu, không có `gap`.
     Kết quả seed 0, 12 tờ: tiêu chí 1 **1361/1361**; tiêu chí 2, 3, 6 đạt cả 12 tờ; tiêu chí 5 vẫn 2
     cặp bong bóng (tờ 7, 9 — mục 2). 5 seed: 50/55 đường vuông góc vào khe, 5/55 đường chéo (đều là tờ
     2 `general26`). 487 test pass, ruff sạch. Ảnh: `stems_fix/tagged_511_diagonal.png` (trái bản gốc,
     phải bản vẽ).
   - Ảnh (vòng 2: "trước" = bản vòng 1, "sau" = bản hiện tại): `stems_fix/tagged_511_{1..4}.png` (11 ca, mỗi ảnh 3 ca; trên là trước, dưới là sau; "trước" = bộ luật
     cũ + bắt đầu ở mực gần nhất). Bố cục tờ khác nhau giữa hai bên vì luật mới đổi số lần rút `rng`.
2. ✅ **XONG 2026-11-04** — **Cặp bong bóng tiếp xúc nhau** (thêm 2026-11-02). Cặp đồng hồ đứng sát nhau (TE …A/B, FO/FT;
   cạnh nhau, chồng nhau hoặc chéo nhau) ở bản gốc **vừa chạm**, nhưng bbox PID2Graph của chúng chồng
   1–5 px (do gán nhãn). Vẽ vòng tròn nội tiếp bbox thì:
   - **đè nhau**: 14 cặp trên 12 tờ (seed 0), 5–18 px. Tiêu chí 3 không bắt được, vì các cặp này đã
     chồng nhau từ bản gốc nên được miễn.
   - **hở nhau**: box rộng hơn cao làm vòng tròn co vào trong box; 2 cặp ở tờ 7, 9 hở 18 và 2 px, cũng
     là 2 cặp trượt tiêu chí 5.

   Sửa: với hai bong bóng nối nhau bằng liên kết dài 0, hoặc có vòng tròn đè nhau, đặt hai vòng **tiếp
   xúc ngoài**. Mỗi vòng dịch một nửa phần chồng (hoặc phần hở) dọc theo đường nối hai tâm; bán kính giữ
   nguyên, phần dịch nhiều nhất khoảng 9 px (đo thực tế: 10.7 px, xem ghi chú dưới). Vòng tròn tính
   **một lần** vào một bảng
   (`node → tâm, bán kính`). Cả bốn chỗ đang tự tính lại từ bbox phải đọc từ bảng đó: chỗ vẽ, chỗ kéo
   ống tới vòng (`_circle_hit` trong `_port_targets`), `_stems`, và bbox annotation / chống đè
   (`_shape_box`). Nếu không, ống sẽ dừng ở vòng tròn cũ. Kiểm: không còn cặp vòng tròn đè nhau quá 1
   nét; tiêu chí 5 hết 2 cặp bong bóng–bong bóng; tiêu chí 1 không trượt thêm cổng bong bóng nào.
   **Kết quả (2026-11-04):** `_bubble_circles` (`page_svg.py`) tính bảng `node → (tâm, bán kính)` một lần
   ngay sau khi biết `kind`; bốn chỗ đọc từ bảng: vẽ vòng tròn, `_circle_hit` trong `_port_targets`,
   `_stems`, và `_shape_box` (bbox annotation + chống đè, qua `_resolve_overlaps`). Quy tắc: hai bong
   bóng nối bằng run dài 0, hoặc nối bằng run mà vòng tròn cách ≤ 2 nét (`_BUBBLE_JOIN_STROKES`), hoặc
   vòng tròn đè nhau quá 1 nét (có nối hay không), thì mỗi vòng dịch một nửa phần chồng/hở dọc đường
   nối tâm để **tiếp xúc ngoài**; bán kính giữ nguyên; lặp tối đa 100 vòng để hàng 3 bong bóng ổn định.
   Cặp không nối và hở không quá 1 nét thì giữ nguyên. Seed 0 trên 12 tờ: 17 cặp được dịch (14 cặp đè
   5–18 px, cặp hở 21 px ở tờ 7, cặp hở 5 px ở tờ 9, cặp hở 3 px ở tờ 2).
   `check_g1` thêm **tiêu chí 7**: không có cặp vòng tròn đè nhau quá 1 nét. Tắt việc dịch thì tiêu
   chí 7 trượt đúng 14 cặp (khớp số đo ban đầu). Sau khi dịch: tiêu chí 1 đạt cả 12 tờ,
   tiêu chí 5 hết 2 cặp bong bóng, tiêu chí 7 0 cặp; seed 1, 2 trên 12 tờ không thêm lỗi nào do mục này.
   492 test pass. Ảnh trước/sau (17 cặp, 3 cặp mỗi ảnh): `docs/experiments/iso_render/tag_pairs/`.
   **Còn mở (có từ trước, không do mục 2):** seed 1, tờ 10: `arrow67` – `general129` hở 4.6 px (tiêu chí
   5); tắt việc dịch vòng tròn vẫn trượt giống hệt.
   **Ghi chú (2026-11-04, sau review):**
   - **Cặp TE 210207A/B (tờ 7) cố ý vẽ tiếp xúc, dù bản gốc hai vòng cách nhau** (~7 px nguồn,
     21 px sau ×S). Khoảng cách ở OPEN100 là do chữ "210207A/B" tràn ra ngoài vòng; bbox PID2Graph
     ôm cả chữ nên chồng nhau và có cạnh nối; V4 đóng khe nên run thành dài 0. Bản vẽ của ta không có
     chữ: giữ khoảng hở thì đồ thị có cạnh mà ảnh không có mực nối (đúng lỗi tiêu chí 5 bắt), tức nhãn
     nhiễu cho mô hình trích đồ thị. Muốn giữ hở thì phải bỏ cạnh khỏi GraphML ở V8 và dò mực ảnh gốc để
     tách cặp "chạm thật" với cặp "dính vì chữ" — không đáng cho 1 cặp / 12 tờ (cặp hở 5 px tờ 9 và
     3 px tờ 2 ở bản gốc vốn sát nhau). Độ dịch lớn nhất đo trên 5 seed: **10.7 px** (cặp này),
     9.3 px (tờ 6 `instrumentation6`/`136`), mọi cặp khác < 9 px.
   - **Hai khóa log mới** trong `Page.log` (`check_g1` in ra ở dòng `log:`): `bubbles_moved` = số bong
     bóng có vòng rời tâm box; `bubbles_unsettled` = số cặp còn phải dời khi hết lượt quét
     (`_BUBBLE_ROUNDS` = 100). Dời một cặp có thể làm đè cặp kế (hàng tag), nên phải quét nhiều lượt:
     hàng 5 tag cần 14 lượt. `unsettled` > 0 nghĩa là vòng lặp bị cắt khi chưa xong; V6 sinh nhiều biến
     thể thì lọc theo khóa này. Seed 0, 12 tờ: `bubbles_moved` 0–10 mỗi tờ (tổng 36),
     `bubbles_unsettled` = 0 cả 12 tờ; 5 seed × 12 tờ đều 0. `_bubble_circles` nhận thêm `stats` và
     `rounds`. Test: `test_bubble_stats_count_moved_tags_and_pairs_left_unsettled` (hàng 5 tag:
     mặc định 0 cặp sót, `rounds=1` còn sót; hai tag xa nhau: 0/0),
     `test_render_page_logs_the_moved_tags`. 494 test pass.
     Ảnh minh hoạ: `docs/experiments/iso_render/tag_pairs/log_counters.png` (trái: tờ 5 thật, vòng cũ
     nét đứt đỏ và vòng đã dời; phải: hàng 5 tag giả, trước khi dời / `rounds=1` / mặc định).
3. Commit mốc "G1 đạt" (cả nhánh hiện còn untracked).
4. ✅ **V5b** — vẽ bình và agitator (xem dưới; xong 2026-11-05).

Nhỏ, làm khi tiện: 5 mũi tên `2_angle`.

## V5b — Vẽ bình `multi` thật + agitator 🤖 (sau G1, trước V6) — 🟢 XONG 2026-11-05 (516 test)

Tách ra từ "Phạm vi lượt này" ở V5. Làm trước V6 vì: V6 sinh và lọc trùng biến thể, nếu bình còn là
hộp thì đổi sau sẽ phải sinh và kiểm lại; và nhiều phần V5 dựa vào việc bình là hộp (ống vòi bình
`_nozzle_links`/`_nozzle_stub`; nối ống vào bình; chống đè).

**Quyết định của user (2026-11-05)**
- **Scope bình**: 2062 + X8100, chọn theo tỉ lệ bbox nguồn (cao/rộng): **≥ 3 → X8100 `column`** (trong
  vocab), còn lại → **2062 `vessel_dished_ends`** (nhãn `other_iso`). 301 và 2063 vẫn `extra`.
- **Agitator**: nhãn COCO theo `fine_name` (`agitator_general` / `_impeller` / `_turbine`); chỉ đặt
  vào 2062 **đứng** (cao/rộng ∈ [1.2, 3)), **không** vào cột hay bình nằm ngang (thường là bộ gia
  nhiệt); chỉ nút nhãn **`tank`** (`agitators.labels`); `p = 0.3`. **GraphML không thêm nút agitator** (PID2Graph không có, agitator không nối ống nào; thêm
  nút làm topology lệch bản gốc khi đánh giá đồ thị); annotation COCO mang **`added: true`** để báo cáo
  tách có/không agitator.

**Quyết định thêm khi code** (user chưa duyệt, đổi được bằng config):
- Bình có cao/rộng < 1 vẽ **xoay 90°** (đầu bầu ở hai đầu trục dài) — bộ gia nhiệt nằm ngang trông
  đúng, bình 2062 đứng kéo méo thì sai. Cột không bao giờ xoay.
- Nút `tank` **không có cổng** (chỉ nối tag) vẫn vẽ thành bình (`allow_empty`); trước đây là hộp
  `other_generic` ("bình cần ≥ 1 cổng"). Chính 5 bình tờ 9 này làm số bình đủ điều kiện agitator
  thành 20 (15 nếu không tính chúng). `general` lớn 0 cổng vẫn là hộp. **User duyệt 2026-11-05.**
- 4 ký hiệu vào `scope: html`: X8100, 2672, C2024, C2027 → **51 template** (taxonomy + sinh lại
  `templates.csv`; SVG không đổi byte, chỉ cột `scope`).

**Sửa sau review (2026-11-05)** — bỏ `min_width`, thêm `labels: [tank]`:
- `min_width = 70` dựa trên nhận định sai ("57/62 px quá hẹp"): 4 nút bị loại (5/`tank55`, 5/`tank108`,
  11/`tank90`, 11/`tank131`) là **bơm đứng** (RCS-PU-102A/B, RHR-PU-120A/B) mang nhãn `tank`, không phải
  bình hẹp; agitator vẽ theo tỉ lệ đường kính bình nên bình hẹp vẫn đặt được (thử p = 1: 4/4 đặt, 0 bỏ).
  Ngưỡng cũng không lọc được bơm: 2/`tank1–3` (CND-PU-16x, rộng 84 px) vẫn lọt. Danh tính gốc (bơm,
  lọc) mất khi vẽ lại thành 2062 nên ảnh vẫn tự nhất quán → bỏ ngưỡng.
- 3 nút `general` lớn từng đủ điều kiện đều **không phải bình**: 1/`general93` (dàn quạt ACC-AC-150,
  mũi tên dòng chảy vẽ xuyên thân, cạnh cánh khuấy), 6/`general39`, 6/`general56` (khung "DETAIL B –
  REACTOR COOLANT PUMP"). Vẫn vẽ 2062 (`other_iso`, vô hại) nhưng không được agitator (nhãn vocab).
- Đủ điều kiện: **20 nút, toàn `tank`** (test `test_real_sheets_..._agitators_their_rule` khóa = 20).
  Trong đó: 6 lò phản ứng tờ 9, 2 bình thật (7/`tank81` ACW-TK-212, 10/`tank192` CVCS-VCT-118), còn lại
  là bơm / lọc / trao đổi ion / tách (vẽ lại thành 2062).
- Ảnh minh hoạ cách V5b làm: `docs/experiments/iso_render/v5b_vessels/v5b_how.png` (luồng V3 → V5,
  hình học `_Vessel`, ống tới viền, đặt agitator; 6 ví dụ gốc ↔ V5b, gồm `general93` không agitator).

**Làm gì (code)**
- V3 (`iso_mapping.py`, `configs/render_iso.yaml`): khóa luật mới `min_aspect` / `max_aspect` (cao/rộng
  nguồn, [min, max)), `allow_empty`, `not_classes`; hai luật `tank` (column rồi 2062) và hai luật
  `general` lớn; `orientations_for(..., aspect, allow_empty)` trả rot 90 cho bình nằm; khối `agitators`
  trong config + `Assignment.agitator` (chọn đều theo lớp, rồi đều template).
- V5 (`page_svg.py`): `_Vessel` vẽ bình **tham số** trên bbox **sau V4** (không phóng méo template: đường
  kính = cạnh ngắn, k = đường kính / 20 mm, nắp elip giữ đúng bézier của BK-2020 ở tỉ lệ k, thân kéo dài
  theo cạnh dài, nên nét luôn vuông; cột thêm 2×3 vạch khay đứt). `_Vessel.target`: đầu ống dò viền bằng
  tia (`_ray_polygon` trên đa giác viền); ống bên sườn nắp (cổng bên trong vùng nắp) kẹp vào thân thẳng
  rồi nối đoạn gấp. `_vessel_stub`: ống vòi bình tính từ viền thật tới cổng ký hiệu (thay `_nozzle_stub`
  cạnh hộp). Kind nội bộ `vessel`; annotation `kind: iso`.
- Agitator (`_place_agitator`): vẽ ở tỉ lệ của bình (cánh = nửa đường kính, như BK-2020) nhưng ≤ 0.7 chiều
  dài bình; trục thẳng đứng trên đường trục, nhô 3 mm khỏi đỉnh bầu; cánh đặt thấp, cách đáy `clearance`
  đường kính (xem "Sửa độ sâu cánh"); né ống vào đỉnh (dời trục ≥ 8 nét, nhưng ≤ 10% bề rộng bình,
  `_AGITATOR_MAX_OFFSET`; xa hơn → bỏ agitator);
  trục không nhô nếu chạm ký hiệu khác; trùng ký hiệu nằm trong bình → bỏ agitator (log
  `agitators_skipped`). Log mới: `vessels`, `agitators`, `agitators_skipped` (thay `multi_as_generic`).
- Annotation agitator: `id = "<node>/agitator"`, `node` = nút bình, `added: true`, bbox = hộp mực (gồm
  đoạn trục nhô). `check_g1` gom theo `node` nên không tính là cặp đè.

**Sửa độ sâu cánh (2026-11-05, sau đối chiếu ISO)** — chuẩn: BK-2020 X8006 ("Jacketed vessel with dished
ends and agitator driven by electric motor", ví dụ ghép bình + agitator duy nhất của ISO 10628-2):
- Đã khớp từ trước: agitator nằm trong bình, gắn từ đỉnh, trục đứng nhô 0.15·d (= 3 mm ở thang 20 mm),
  cánh ≈ 0.5·d (X8006: 0.51; ta: 0.505).
- Lệch: ta dùng nguyên template (trục chỉ dài ≈ bề rộng cánh) nên cánh treo sát nắp — đáy cánh ở 0.40 /
  0.30 / 0.21 chiều cao (cao/rộng 1.5 / 2.0 / 2.8); X8006: 0.67, tức cách đáy 0.50·d.
- Sửa: template giữ tỉ lệ bình, đặt đáy cánh cách đáy viền `clearance`·d, `clearance` rút đều trong
  `agitators.clearance` = [0.35, 0.65] ở V3 (`Assignment.clearance`, mặc định 0.5); V5 vẽ thêm đoạn trục từ
  điểm nhô trên đỉnh xuống, chồng vào trục của template (`_Agitator`: `pl`, `x`, `top`, `joint`). Bình
  quá thấp (cánh không xuống được) → treo template từ đỉnh như cũ. Bbox COCO phủ cả trục (hộp mực thật như
  P&ID thật; crop sẽ cao hơn prototype Stage 2 — đúng dạng gặp ngoài thực tế).
- Kiểm: cao/rộng 1.5, clearance 0.5 → đáy cánh 0.67 (trùng X8006); 4 test mới (độ sâu theo 3 giá trị
  clearance + trục liền nét; bình thấp; clearance rút trong khoảng). G1 12 tờ × seed 0–2: ~~mọi tiêu chí tự động
  đạt cả 36 lượt~~ **ghi sai** — thực tế 32/36 (tờ 10 seed 0–2 tiêu chí 5, tờ 11 seed 1 tiêu chí 1; xem lần 2
  dưới); agitator seed 0/1/2 = 7 / 8 / 3, 0 bị bỏ. Ảnh: `v5b_vessels/agitator_depth.png` (X8006 ↔
  sau sửa), `v5b_how.png` đã vẽ lại.
- **Để sau (user, 2026-11-05)**: motor M trên đầu trục như X8006 (C0082 `motor_electric`, thuộc vocab, đang
  `extra`) — thêm nhãn COCO mới nên cần quyết riêng. Gắn được thẳng vào `_Agitator.top` (đầu trục), không phải
  sửa lại phần đặt cánh.
- Không làm: agitator gắn hông / gắn đáy (ISO/BK-2020 và OPEN100 đều không có ví dụ; bình chủ là 2062 đứng
  hẹp); "trộn trên ống" là ký hiệu khác (X2672 in-line rotary mixer, X2673 static mixer — lưu ý X2672 ≠ 2672
  agitator của ta, lấy từ SHT7).
- Khác OPEN100 nhưng không phải lỗi: độ sâu nắp (OPEN100 elip 0.17–0.2·d hoặc bán cầu; ta 0.125·d theo
  2062), vạch tiếp tuyến ở mối nắp–thân (ISO 2062 không có), chân / váy đỡ (X8002–X8005, ngoài scope).

**Sửa sau đối chiếu thực tế (2026-11-05, lần 2)** — ba sửa, user duyệt:
- **Độ sâu cánh**: `clearance` (đo tới **mép dưới** cánh) [0.35, 0.65] → **[0.25, 0.65]**. Thiết bị thật
  đặt **tâm** cánh cách đáy ≈ T/3 (cấu hình khuấy chuẩn; 0.15–0.5·T tuỳ loại), nên khoảng cũ đặt cánh
  hơi cao so với thực tế (dù khớp X8006). Không có quy định ISO về độ sâu; nguồn là X8006 + tài liệu thiết
  kế khuấy trộn, chưa đo trên tập bản vẽ thật nào (OPEN100 và SynthPID không có agitator).
- **Trục ở tâm**: trước đó trục né ống đỉnh bằng cách dời tới chỗ trống gần nhất — 5/18 agitator lệch
  4–20% bề rộng (7/`tank81` 20%: trông như gắn lệch). Nay lệch tối đa **10% bề rộng**
  (`_AGITATOR_MAX_OFFSET`), xa hơn → bỏ (`agitators_skipped`). Sau sửa: 15 agitator, lệch lớn nhất 6.6%,
  3 bị bỏ (5/`tank55`, 5/`tank108` — hai bơm đứng — và 7/`tank81`: ống vào sát tâm đỉnh).
- **Nút `general` nhỏ 1 cổng → cặp mặt bích (511), không còn G5**: trong OPEN100 có 55 nút loại này (29
  cạnh `tank`, 11 cạnh `valve`, 8 cạnh `general`, 6 cạnh `crossing`, 1 cạnh `arrow`); xem từng nút: **53
  là cặp mặt bích ở đầu ống** (vòi bình, ống xả / thông hơi sau van), 1 bộ phá chân không (2/`general25`), 1
  giống phễu (1/`general45`) — hai nút này cũng thành 511 (gần đúng nhất, không đáng luật riêng). Trước
  đó luật G5 vẽ chúng thành bộ ngăn lửa / van thở / bẫy hơi… — sai, nhất là ở vòi đáy bình (vd. đáy cột
  0/`tank67`). Luật mới `{label: general, size: small, ports: ["1"], groups: [G3], classes:
  [flanged_connection]}` đứng trước luật G5; taxonomy 511 `placement` `pipe` → **`pipe_or_end`** (để một
  cổng hở). G5 vẫn có qua bong bóng 1 cổng được thay (`replace`, p = 0.3).
- **`check_g1` tiêu chí 1**: cửa sổ ±1 px lấy quanh `round(y)` nên bỏ sót mực khi cổng ở toạ độ lẻ ~.5:
  tờ 11 seed 1, đường tag vào khe hai mặt bích (`general83`, `general102`) dừng đúng trục ống ở y = 3755.51,
  hàng 3755 chỉ phủ 51% (xám 136 > ngưỡng 128), mực đậm ở hàng 3754. Hình đúng (đã xem ảnh). Sửa: cửa sổ
  gồm mọi pixel có phần nằm trong 1 px quanh cổng (`floor(y − 1)` … `ceil(y + 1)`), luôn chứa cửa sổ cũ.
- **Đính chính**: lần sửa độ sâu cánh ghi "G1 đạt cả 36 lượt" là **sai** — rút `clearance` dùng thêm một
  số `rng` nên bố cục cả tờ đổi (tờ 11 seed 1: 8166 → 8489 px), làm lộ hai lỗi trên mà không được đọc kỹ.
- Kiểm: 516 test (thêm: 511 ở đầu ống — tổng hợp và 12 tờ × 3 seed = 165 nút; trục lệch quá 10% → bỏ;
  clearance 0.25). G1 12 tờ × seed 0–2: **33/36 đạt**; 3 lượt trượt đều là tiêu chí 5 tờ 10, cặp
  **mũi tên–`general`** đã biết (seed 0 `arrow7–general135`, `arrow8–general180` 1.8 px; seed 1
  `arrow67–general129` 4.6 px; seed 2 `arrow8–general180` 1.8 px) — không liên quan ba sửa này (bỏ cả ba
  vẫn trượt y hệt), **còn mở**. Agitator seed 0/1/2 = 6 / 6 / 3, bị bỏ 1 / 2 / 0. Ảnh `v5b_how.png` vẽ lại
  (panel c: ống đỉnh lệch tâm, trục vẫn ở tâm; ví dụ 5/`tank55` giờ là ca bị bỏ; đáy cột có mặt bích).

**Kết quả lần đầu** (12 tờ; seed 0, trước sửa review): 47 bình (3 `column`, 44 `vessel_dished_ends`), 7 agitator (3 general, 3
impeller, 1 turbine; 0 bị bỏ), 32 ống vòi bình. Seed 1–2: 5 và 1 agitator (p = 0.3 trên ~20 bình
đủ điều kiện, dao động lớn → cần V6 sinh nhiều biến thể mới đủ mỗi lớp agitator). Tiêu chí 1, 2, 3, 6, 7
đạt cả 12 tờ × seed 0–2 (3 lượt đo đầu; sau khi agitator né ký hiệu khác thì hết lỗi tiêu chí 3).
Tiêu chí 5 còn các cặp **mũi tên–`general`** ở tờ 10 (`arrow7–general135`, `arrow8–general180` hở
1.8 px; seed 1 `arrow67–general129` 4.6 px đã ghi từ trước) — không liên quan bình (chưa đo bản gốc
cùng seed để xác nhận). Seed 3 tờ 6: 1 cổng van an toàn + tag trượt tiêu chí 1 (`valve60`, không
liên quan bình). Ảnh: `docs/experiments/iso_render/v5b_vessels/` (`vessels_seed0.png`: 47 bình;
`vessels_seed3.png`: các bình có agitator).

**Kết quả sau sửa review** (2026-11-05, 12 tờ × seed 0–2 = 36 lượt `check_g1`): **mọi tiêu chí tự
động đạt cả 36 lượt** (số dùng `rng` đổi nên bố cục khác lần trước; các cặp mũi tên–`general` tờ 10 không
còn hiện ở 3 seed này, chưa phải là đã sửa). Agitator: seed 0/1/2 = 6 / 8 / 3, 0 bị bỏ.

**Hạn chế đã biết**: ống đi xuyên bbox bình (bản gốc có; `general77` tờ 6) vẫn hiện xuyên thân bình vì bình
không tô nền; mũi tên nằm trong bbox bình (tờ 1 `general93`) vẫn vẽ trong thân; chỉ có 3 cột nên lớp `column`
ít mẫu (V6 phải bù bằng nhiều biến thể); agitator chỉ ~6/seed và dồn về tờ 9 (6 lò phản ứng giống nhau).

## V6 — Sinh nhiều biến thể + lọc trùng 🤖 (~0.5 ngày) — 🟢 XONG 2026-11-05 (533 test)

- Nguồn biến thể: phép gán ký hiệu (V3), ánh xạ trục (V4), `w ∈ [2, 4]·S/3` (S = `render_iso.layout.scale`), `gap_scale` của V4, kích thước ký hiệu ±15%.
- Lọc trùng: aHash + dHash (tự viết, 📎 `diversity.py`) — Hamming < ngưỡng với ảnh cùng gốc → loại;
  chữ ký phép gán trùng → loại.
- Mặc định ~30–40 biến thể/gốc (SynthPID: lợi ích bão hòa ~400 ảnh).

**Đã làm** (chi tiết, số liệu, ảnh: `docs/experiments/iso_render/v6/REPORT.md`; lệnh `scripts/make_variants.py`):
- Code: `src/utils/diversity.py` (aHash/dHash tự viết, `Deduper`), `src/utils/iso_variants.py`
  (`derive_seed`, `plan_variant`, `render_variant`, `generate_sheet`), khối `render_iso.variants` trong
  `configs/render_iso.yaml`; không sửa V3/V4/V5. `scripts/check_g1.py` tách `evaluate()` để dùng làm bộ lọc.
- **RNG độc lập**: mỗi giai đoạn (`assign`, `layout`, `style`) một seed dẫn xuất bằng hash từ (base seed, tờ,
  chỉ số biến thể, tên giai đoạn). `w` ~ U[2, 4] (qua bản sao `cfg`), hệ số cỡ ~ U[0.85, 1.15] một hệ số cả tờ
  (qua bản sao `source` chỉ nhân hộp ký hiệu đặt theo cổng + mũi tên; bình/hộp/bong bóng không nhân vì vẽ
  trên hộp V4 — nhân hết làm tiêu chí 3 trượt giả).
- **Thứ tự lọc**: chữ ký phép gán → lỗi V5 → `Page.log` (`bubbles_unsettled`, `unmatched_ports` > 0) → aHash+dHash
  (16×16, trùng khi **cả hai** < 8 bit) → G1. `loose_ends`, `agitators_skipped`, `shrunk` không lọc.
- **Tiêu chí 5 (mũi tên–ký hiệu hở): LOẠI** (chờ user duyệt; đổi bằng `variants.g1.flag`). Họ lỗi này không chỉ ở
  tờ 10: tờ 4 (20 ca), 2 (10), 1 (3) và tờ 10 (27).
- `check_g1` tiêu chí 2: bộ đo 1× cũ đọc nét chẵn thiếu 1 px và nét phân số lệch (trượt giả với w liên tục) →
  đo ở 4× trên 8 ô nhiều mực nhất, ngưỡng 1 px giữ nguyên.

**Kết quả** (base_seed 0, 12 tờ, mục tiêu 36/tờ, tối đa 60 lần thử): 552 lần thử → **413 giữ** (26–36/tờ), loại 66
trùng ảnh (khoảng cách 3–7 bit) + 73 G1 (60 tiêu chí 5, 13 tiêu chí 1); 0 trùng chữ ký, 0 lỗi, 0 vì `Page.log`.
0.70 s/biến thể giữ, ~308 KB/biến thể (PNG 207 + SVG 81 + JSON 29 KB), 127 MB cả bộ, 52 s với 10 tiến trình.
G1 theo w ∈ {2, 2.5, 3, 3.5, 4} × cỡ {0.85, 1, 1.15} (36 lượt/ô): tiêu chí 2, 3, 6, 7 đạt hết; tiêu chí 1 trượt
nhiều ở w = 2 (8/36 ở cỡ 1.15, cổng `loose` của van thân chéo); tiêu chí 5 trượt nhiều ở cỡ 0.85 (5–8/36).
Độ phủ: agitator 180 (124 biến thể có ≥ 1), `column` 108 (chỉ 3 nút nguồn), `flanged_connection` 3964, G5 ngoài vocab
nằm trong `other_iso`. Dữ liệu ở `data/iso_open100/variants/sheet<i>/v<NNN>.{png,svg,json}` (gitignore).

**Hạn chế**: hash 16×16 loại cả biến thể "cùng bố cục, khác ký hiệu"; chỉ 12 topology; số biến thể 26–36/tờ không đều.

## V7 — Làm suy giảm toàn ảnh 🤖 (~1 ngày)

- Chép (không import) `degrade()` của `scripts/render_pid.py` (noise, blur, pixelate) + thêm dilation/erosion,
  nghiêng nhẹ ±1°, JPEG, ánh sáng không đều.
- Giới hạn theo `w`: kernel erosion ≤ `w/2`, blur σ ≤ `w/2`.
- **Kiểm tra chi tiết còn sống**: với mỗi template nhóm van (`template_iso_svg/`, G1/G2), vẽ đơn lẻ ở kích thước đích, suy giảm
  ở mức tối đa, rồi so NCC với mọi template cùng nhóm — phải vẫn gần template của chính nó nhất.
  Template nào trượt → hạ tham số tối đa.

## V8 — Xuất dataset + nối vào pipeline 🤖 (~0.5 ngày)

- Data root riêng `data/iso_open100/` với layout giống `data/`: `canonical/{images/,
  annotations.coco.json, classes.yaml}`, `graphml/`, `splits.json`. Chạy pipeline trên tập này bằng
  `PID_DATA_ROOT=data/iso_open100` + override `oneshot.symbols_dir` / `oneshot.vocab_csv` về
  `data/raw/...` — **không bao giờ** ghi vào `data/canonical/` của DigitizePID.
- **Chia theo bản vẽ gốc** (không theo biến thể): nếu chỉ dùng làm test thì cả 12 gốc vào test; nếu
  train Stage 1 thì tách gốc train/test, không gốc nào ở cả hai bên.
- Kiểm tra layout khớp `DataPaths` để `detect_infer` / `fgc_infer` / `evaluation` đọc được.
- Ghi `annotations.coco.json` (bbox = hộp mực từ `Page.annotations` của V5, `category = fine_name`
  hoặc `other_iso`/`other_isa`/`other_generic`) + GraphML nhãn mới (chuyển từ V5, 2026-10-31).
- Thống kê: histogram lớp, số ký hiệu/ảnh, tỉ lệ `other_*`.
- [ ] **Toạ độ bong bóng trong GraphML lấy từ vòng tròn đã dời** (`_bubble_circles`, cũng là bbox
      annotation), không lấy từ box V4: cặp tag tiếp xúc bị dời tới 10.7 px ("Trước V6", mục 2).
      Kiểm: tâm node bong bóng trong GraphML trùng tâm bbox COCO của nó.

## V9 — (Tùy chọn) Thêm khung xương 🤖

`Dataset PID` (500, topology tổng hợp) và `PID2Graph Synthetic` (250) trong zip — chỉ khi cần thêm
dữ liệu train Stage 1. Không dùng làm test thay OPEN100 (topology không thật).

---

## Thứ tự và thời gian

```
V0 🧑 ─► V1 (van trước) ─► V2 ─► V3 (nhóm van) ─► V4 ─► V5 ─► ⛔ G1 (≈ 4–5 ngày làm việc)
                                                                │
                                   đạt ─► V1/V3 các nhóm còn lại ─► V6 ─► V7 ─► V8 (≈ 3–4 ngày)
                                   trượt ─► dừng, ghi kết quả âm vào HANDOFF §4
```

**Cập nhật 2026-11-02 — G1 ĐẠT.** V1/V3 đã phủ đủ 5 nhóm ngay trong vòng G1. Thứ tự tiếp theo:
`sửa _stems ─► cặp bong bóng tiếp xúc ─► commit mốc G1 ─► V5b (bình + agitator) ✅ ─► V6 ─► V7 ─► V8`.

Không chặn `PLAN_STAGE2_ONESHOT.md`: R1a/R1b (Colab train encoder) không cần nhánh này. Đề xuất làm
sau khi B6a/B6b code xong, hoặc xen kẽ khi chờ Colab.

## Rủi ro đã biết

| Rủi ro | Giảm thiểu |
|---|---|
| Ảnh quá sạch, khác bản scan thật | V7 + báo cáo trung thực là tập ISO giả lập |
| Mỗi lớp một cách vẽ → điểm one-shot cao giả | Biến thể kích thước/nét (V6), dùng lại `SubpartWarp` (B1) cho ký hiệu; ghi hạn chế |
| Chỉ 12 gốc, ít pump (16) / tank (35) | Chấp nhận; báo cáo số lượng theo lớp |
| Ký hiệu ISO to hơn chỗ trống | Luật fallback V3 |
| Cổng tự đề xuất từ stub sai | Đã duyệt preview cổng ở V1 (2026-10-29); `templates.csv` cột `ok` 62/62 |
| License | Code tự viết; BK-2020-04 CC BY-SA 4.0 (trích dẫn); PID2Graph trích dẫn bài gốc |
