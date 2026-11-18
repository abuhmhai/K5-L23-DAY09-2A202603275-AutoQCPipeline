# Báo cáo Day 09 — Minh Hải, 2A202603275

## 1. Kết luận trước

Pipeline kết luận **Hold** (mã thoát 10). Vào 224 ảnh (`data/raw`); bước nhận loại 22 ảnh; xuất 202/224 ảnh trong `submission/export/clean_v1/` (`manifest.json`: `n_images` 202, `decision` Hold). Đây là bản nháp, chưa được phát hành.

Lý do: sau khi máy sửa hết lỗi bắt buộc và người duyệt mô phỏng xem 80/125 ảnh trong hàng đợi, vẫn còn 38/202 ảnh có cảnh báo chưa được xem (`08_summary.json`: `warn_ratio` 0.1881 ≥ ngưỡng 0.10). Không còn lỗi bắt buộc (`n_errors` 0).

## 2. Lỗi và phân việc

Đếm từ `02_error_list_v1` (211 dòng, mẫu số 202 ảnh đã nhận; 22 dòng đầu là ảnh đã loại ở bước nhận). Cột "ai sửa" theo việc thật sự xảy ra trong `03_fixes.csv` và `01_intake.csv`.

| Mã lỗi | Số dòng / số ảnh | Ai sửa | Vì sao |
|---|---:|---|---|
| POSSIBLE_MISSING | 53 / 52 | người | Model thấy vật, nhãn không có box. Máy chỉ cảnh báo. |
| POSSIBLE_FALSE_POS | 29 / 27 | người | Box nhãn không khớp dự đoán nào. Model có thể sai. |
| POSSIBLE_LOOSE_BOX | 28 / 24 | người | Đúng lớp nhưng IoU thấp. |
| POSSIBLE_WRONG_CLASS | 18 / 18 | người | Cùng chỗ, khác lớp. |
| EMPTY_LABEL | 12 / 12 | người | File rỗng có thể là ảnh nền hoặc thiếu nhãn. |
| DUP_IMAGE_EXACT | 8 / 8 | máy | Trùng byte. Intake xóa ảnh sau. |
| BOX_OUT_OF_IMAGE | 8 / 8 | máy | `clip_box` cắt về [0, 1] (8 dòng trong `03_fixes.csv`). |
| DUP_BOX | 7 / 7 | máy | `drop_line`. |
| MISSING_LABEL_FILE | 6 / 6 | người | Máy ghi `needs_label`, không tạo file rỗng. |
| BAD_CLASS_ID | 6 / 6 | máy | `drop_line`. |
| BOX_TINY | 6 / 4 | người | Cảnh báo box nhỏ, dưới 12 px. |
| LEAK_TEST | 5 / 5 | máy | Trùng test/val. Intake xóa. |
| DUP_IMAGE_NEAR | 4 / 4 | máy | Phash ≤ 12. Intake xóa. |
| BOX_ZERO | 4 / 4 | máy | `drop_line`. |
| CLASS_RARE | 4 / 1 (`<dataset>`) | máy | Đếm box id lạ (−1, 4, 7, 93). Hết sau khi các dòng đó bị xóa; không vào hàng đợi. |
| BAD_SIZE | 3 / 3 | máy | Cạnh ngắn < 250 px. Intake xóa. |
| BAD_FORMAT | 3 / 3 | máy | `drop_line`. |
| BOX_HUGE | 3 / 3 | người | Box phủ > 0.95 ảnh. |
| CORRUPT_IMAGE | 2 / 2 | máy | Không mở được. Intake xóa. |
| BAD_CLASS_NAME | 2 / 1 (`data.yaml`) | máy | `rename_classes` về car, person, truck, bus. |

`05_review_queue.csv` xếp theo điểm giảm dần. `img_0020` đứng đầu (11.524) vì cộng `FIXED_DROP_LINE` (2) + `POSSIBLE_WRONG_CLASS` (3) + `POSSIBLE_MISSING` (2), rồi nhân thêm độ tin của model. Sáu ảnh `NEEDS_LABEL` đứng ngay sau với điểm 10: trọng số cao nhất trong bảng, vì chỉ người gán được nhãn mới. Với ngân sách 80 ảnh, người xem được 94/123 dòng ngữ nghĩa (`POSSIBLE_*` ở `04_error_list_v2`) trên 74 ảnh trong 80 ảnh đầu. 29 dòng còn lại (16 box lỏng, 11 false positive, 2 thiếu nhãn) nằm ngoài ngân sách và vẫn tính vào tỉ lệ cảnh báo, nên cổng ra Hold.

## 3. Calibration (2 ngưỡng)

Sweep trong `submission/calibration/calibration_report.md`. Ô là precision / recall / F1.

| Tham số | Khởi đầu | Đã chọn | Precision / Recall (trước → sau) | Lý do |
|---|---|---|---|---|
| `duplicates.phash_dist` | 2 | 12 | NEAR 1.00/0.50 → 1.00/0.67. EXACT giữ 1.00/1.00. | Xóa ảnh nhầm đắt hơn bỏ sót. 12 vẫn precision 1.00 trên calib (4/6 near-dup) và bắt cặp raw Hamming 10. |
| `second_opinion.conf` | 0.25 | 0.6 | WRONG 0.56/0.83 → 0.62/0.83. MISSING 0.50/1.00 → 0.57/1.00. LOOSE 0.29/0.40 → 0.20/0.20. FALSE_POS giữ 0.31/1.00. | Giữ recall thiếu nhãn = 1.00 và recall nhầm lớp = 0.83, bớt cờ giả. FALSE_POS không đổi vì dùng `fp_conf` cố định. |

Chỗ calibration đánh lừa: F1 near-dup cao nhất ở `phash_dist` 16 (NEAR 1.00/0.83/0.91) và precision trên 39 ảnh calib vẫn là 1.00. Calib không có ca `LEAK_TEST`. Trên raw, Hamming 16 gộp `img_0011` (vận động viên tennis) với `img_0008` (hai người trượt tuyết) và kéo `LEAK_TEST` từ 5 lên 11 ảnh. Ngược lại, `img_0186` và `img_0061` (Hamming 10) là cùng một ảnh người ngồi ghế đá, khác kích thước; ngưỡng 8 bỏ sót cặp này, ngưỡng 12 bắt được. `calib_038` cách ảnh gốc 18 bit (mức biến đổi 4) không có ngưỡng nào bắt an toàn, vì raw có cả một cụm cặp khác cảnh ở Hamming 18–20.

## 4. Cửa chặn và điều pipeline không bắt được

(a) `python -m pipeline.run --budget 0 --reports work/rep_nb --export-dir work/exp_nb` kết luận **Rework**, mã thoát 20. Lý do in ra: còn 6 lỗi bắt buộc trên 202 ảnh (đủ 6 `MISSING_LABEL_FILE`, máy không bịa nhãn, ngân sách 0 nên không ai gán). Log ghi không xuất dataset; `work/exp_nb` không tồn tại.

(b) Hai điểm mù có ảnh chứng:

- Box lỏng nhẹ: `img_0068` dòng 3, lớp person, IoU 0.523 với dự đoán conf 0.854. Ngưỡng `iou_loose` là 0.5 nên pipeline im, dù box đã lệch.
- Trùng biến đổi mạnh: `calib_038` cách `calib_024` 18 bit và không bị cờ ở `phash_dist` 12. Kéo ngưỡng lên 16 để bắt `calib_037` (16 bit) thì raw mất ảnh tennis ở trên.

## 5. Train: raw vs clean (tùy chọn, chấm riêng)

Chạy local trên GPU Apple (MPS), cùng script notebook: `yolo26n.pt`, 20 epoch, `imgsz` 640, `batch` 16, seed 0 và 1. Val là `gold_val` (40 ảnh), test là 100 ảnh. Raw 224 ảnh train, clean 202 ảnh (bản Hold). Bảng từ `submission/results/comparison.md`:

| biến thể | #ảnh train | số seed | map50 (TB ± độ lệch) | map50_95 (TB ± độ lệch) | precision (TB ± độ lệch) | recall (TB ± độ lệch) |
|---|---|---|---|---|---|---|
| raw | 224 | 2 | 0.716 ± 0.024 | 0.511 ± 0.024 | 0.729 ± 0.124 | 0.642 ± 0.063 |
| clean | 202 | 2 | 0.707 ± 0.014 | 0.505 ± 0.020 | 0.755 ± 0.047 | 0.605 ± 0.016 |

Chênh mAP50 (clean − raw) = **−0.009**. Độ lệch giữa hai seed là 0.024 (raw) và 0.014 (clean). Chênh lệch nhỏ hơn nhiễu seed, nên **chưa kết luận được** clean tốt hơn raw.

Hai giả thuyết:

- Ultralytics đã bỏ 11 ảnh raw hỏng hoặc nhãn sai (class 93, id âm, file cụt, box tràn) và chỉ train 213/224 ảnh. Phần "raw" vì vậy đã được trainer tự lọc một phần. Cách kiểm: so số ảnh trong log train với 224 file trên đĩa, và train lại raw sau khi chỉ bỏ đúng các ảnh đó.
- Clean vẫn là bản Hold: 38/202 ảnh còn cảnh báo chưa người xem, nhãn gốc là nhãn máy. Test chỉ 100 ảnh nên mAP50 nhảy 0.01–0.02 giữa hai seed. Cách kiểm: duyệt hết hàng đợi rồi train lại, và thêm seed để ước lượng nhiễu.

## 6. Quyết định và ghi chú AI

(a) Giữ **Hold**. Để thành Release cần người xem nốt 38/202 ảnh còn cảnh báo (phần đuôi của 125 ảnh trong hàng đợi, vượt ngân sách 80) và xử lý hoặc chấp nhận các cảnh báo đó để `warn_ratio` xuống dưới 0.10. Không còn lỗi bắt buộc, nên không cần Rework trừ khi người duyệt để lại dòng nhãn hỏng.

(b) Có dùng AI để viết 4 module.

- Prompt chính: viết `label_format.run`, `gate.decide`, `autofix.fix`, `second_opinion.run` đúng docstring và test sẵn; đọc ngưỡng từ `cfg`; không sửa file đã ghim.
- Cách kiểm: `pytest` 79 passed; chạy pipeline; mở cặp ảnh bị gắn cờ trùng.
- Một lần sai: chọn `phash_dist` 16 vì F1 calib cao nhất. Đối chiếu ảnh raw thì cặp Hamming 16 là hai cảnh khác nhau. Đổi lại 12.
