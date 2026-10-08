# Checklist Lab 22 — DPO/ORPO Alignment (VS Code + extension Colab, GPU T4)

Đánh dấu `[x]` khi xong từng mục. Thứ tự dưới đây là thứ tự nên làm.
Phần bắt buộc = 100 điểm (NB0 → NB4 + REFLECTION + nộp bài). Bonus tối đa +20.

> **Cách chạy:** mở notebook ngay trong VS Code trên máy, kernel là **Colab server (T4)**.
> GPU của máy (RTX 3050 Ti 4 GB) **không đủ** cho Qwen3-4B (cần ≥ 12 GB), nên đừng chọn kernel Python local.
>
> **Nhớ 4 điều:**
> 1. Notebook `.ipynb` nằm trên máy → output tự lưu khi bạn `Ctrl+S`.
> 2. Nhưng mọi file kết quả (ảnh, json, parquet, mô hình) nằm trên **ổ đĩa của Colab server** (`/content/lab22`)
>    và **mất hết** khi server bị ngắt → phải đẩy về repo trước khi ngắt (mục 6).
> 3. Chỉ file **đã commit lên GitHub** mới được chấm.
> 4. Không bao giờ commit `.env`, khoá API, token GitHub hay trọng số mô hình (`.safetensors`, `models/`).

---

## 0. Chuẩn bị (~10 phút)

### 0.1 Repo GitHub
- [x] Tạo repo **public** trên GitHub (vd: `K4-L3-Track3-Day22-NguyenHaiNam-2A202602476-DPO-ORPO-Alignment`).
- [x] Trong thư mục lab trên máy: `git remote add origin <url>` (nếu chưa có) → `git push -u origin main`.

### 0.2 Mở notebook trong VS Code và nối kernel Colab
- [ ] Cài extension **Google Colab** (publisher: Google) và **Jupyter** trong VS Code.
- [ ] Tạo bản sao để nộp (giữ file gốc sạch):
  ```powershell
  Copy-Item colab/Lab22_DPO_T4.ipynb colab/Lab22_DPO_T4_executed.ipynb
  ```
- [ ] Mở `colab/Lab22_DPO_T4_executed.ipynb` → **Select Kernel** (góc phải trên) → **Colab** → **New Colab Server**
      → đăng nhập Google → chọn **GPU → T4**.
- [ ] Giữ VS Code mở và máy không sleep suốt lúc chạy (đặt Power → Sleep: Never khi cắm sạc).
- [ ] Nhấn `Ctrl+S` thường xuyên để output được ghi vào file `.ipynb` trên máy.

### 0.3 Phần A — Setup
- [ ] Chạy cell cài đặt đầu tiên (`COMPUTE_TIER = "T4"`). Giữ nguyên, **không** cần API key.
  - Nếu từng bị OOM: thêm `os.environ["MAX_LEN"] = "512"` vào cell này.
- [ ] Chạy cell `!pip install ...` (vài phút). Nếu được yêu cầu restart → **Restart** kernel (thanh trên notebook) rồi chạy lại các cell phần A, trừ cell pip.
- [ ] Chạy cell tạo `/content/lab22` và các cell `%%writefile` ghi package `lab22/`.
- [ ] Kiểm tra GPU: thêm 1 cell `!nvidia-smi` → thấy `Tesla T4`.

### 0.4 Chuẩn bị đẩy kết quả từ Colab server về GitHub
Trong VS Code, `drive.mount` và `files.download` của Colab có thể không chạy (chúng cần giao diện trình duyệt),
nên cách chắc chắn nhất là **commit + push thẳng từ Colab server** lên repo, rồi `git pull` về máy.

- [ ] Repo đã public trên GitHub (`namhaing/K4-L3-...`) để server clone được.
- [ ] Tạo **GitHub fine-grained token**: GitHub → Settings → Developer settings → Fine-grained tokens →
      chỉ chọn repo lab này → quyền **Contents: Read and write** → hạn 7 ngày. Lưu tạm, **không** dán vào notebook.
- [ ] Thêm 1 cell **ngay sau cell `%%writefile .../modeling.py`** (trước tiêu đề NB0). Chạy 1 lần mỗi khi có server mới:
  ```python
  import getpass, subprocess
  from pathlib import Path

  REPO_URL = "https://github.com/namhaing/K4-L3-Track3-Day22-NguyenHaiNam-2A202602476-DPO-ORPO-Alignment.git"
  TOKEN = getpass.getpass("GitHub token: ")  # dán token vào ô hiện ra, không in ra
  PUSH_URL = REPO_URL.replace("https://", f"https://x-access-token:{TOKEN}@")
  !rm -rf /content/repo && git clone -q {REPO_URL} /content/repo
  !git -C /content/repo config user.name "Nguyễn Hải Nam" && git -C /content/repo config user.email "namhaii631@gmail.com"

  # Chỉ các file kết quả nhỏ; .gitignore của repo tự chặn trọng số mô hình.
  FILES = ["submission/screenshots", "data/eval", "data/pref/train.parquet", "data/pref/eval.parquet",
           "adapters/sft-mini/adapter_config.json", "adapters/dpo/adapter_config.json",
           "adapters/dpo/dpo_metrics.json", "adapters/dpo/split.json",
           "adapters/variants/variants_summary.json", "adapters/grpo/grpo_metrics.json"]


  def backup(msg="lab22: kết quả từ Colab"):
      def git(*args):
          r = subprocess.run(["git", "-C", "/content/repo", *args], capture_output=True, text=True)
          out = (r.stdout + r.stderr).replace(TOKEN, "***").strip()
          if out:
              print(out)
          return r.returncode

      present = [f for f in FILES if (Path("/content/lab22") / f).exists()]
      if not present:
          print("Chưa có kết quả nào để sao lưu.")
          return
      subprocess.run(["cp", "-r", "--parents", *present, "/content/repo/"], cwd="/content/lab22", check=True)
      git("add", "-A", "submission", "data", "adapters")
      git("commit", "-qm", msg)
      git("pull", "-q", "--rebase")
      if git("push", "-q", PUSH_URL, "HEAD:main") == 0:
          print(f"✓ Đã push: {present}")
  ```
  Token chỉ nằm trong bộ nhớ kernel, không lưu vào `git remote`, và bị che thành `***` nếu git in lỗi.
- [ ] **Sao lưu** = thêm 1 cell `backup()` ở cuối mỗi NB rồi chạy → thấy `✓ Đã push: [...]`.
- [ ] Trên máy: `git pull` để kéo kết quả về.

---

## 1. NB0 — Tự viết DPO loss (~10 phút, 10 điểm)

- [ ] Chạy cell import + §1 (log-prob của một câu trả lời).
- [ ] Điền hàm `my_dpo_loss` (chỗ `# TODO`):
  - margin = `beta * ((pc - rc) - (pr - rr))`
  - loss = `-torch.nn.functional.logsigmoid(margin)` rồi lấy `.mean()`
- [ ] Chạy cell kiểm tra → thấy `✓ Khớp tham chiếu: ...`. Nếu báo `Chưa cài` / lỗi assert → kiểm tra dấu trừ và β.
- [ ] Chạy §3 → `loss at init` ≈ `0.6931` = log 2.
- [ ] Chạy §4 (trọng số gradient), §5 (likelihood displacement), §6 (4 biến thể).
- [ ] **Trả lời câu hỏi (4 điểm):** thêm 1 cell **Văn bản** ngay sau §5, viết 3–5 câu:
  > Vì sao margin tăng được trong khi log-prob của `chosen` giảm?
  - Ý chính: loss DPO chỉ phụ thuộc **hiệu** `(log-ratio chosen) − (log-ratio rejected)`. Nếu `rejected` giảm nhanh hơn `chosen` thì margin vẫn tăng, loss vẫn giảm (kịch bản B ở §5 có loss y hệt kịch bản A). DPO không có số hạng nào giữ `chosen` đi lên → cần nhìn đường `rewards/chosen` ở NB3; RPO thêm NLL(chosen) để chống hiện tượng này.
- [ ] (Ghi chú cho REFLECTION) Trả lời ngắn câu hỏi cuối NB0: vì sao DPO gốc dễ thiên vị độ dài (tổng log-prob của câu dài âm hơn), SimPO/ORPO xử lý bằng cách chuẩn hoá theo số token.

---

## 2. NB1 — SFT-mini (~15–25 phút, 8 điểm)

- [ ] Chạy cell import → in `C.summary()`, không lỗi `assert torch.cuda.is_available()`.
- [ ] §1 Nạp mô hình 4-bit + LoRA → in số `Trainable params`.
- [ ] §2 Tải 1.000 mẫu VN Alpaca → in được 1 đoạn `text` mẫu.
- [ ] §3 Huấn luyện (chờ ~15–20 phút).
- [ ] Cell vẽ loss → ảnh `02-sft-loss.png` **đi xuống**. Nếu không giảm: dừng lại kiểm tra GPU/dữ liệu.
- [ ] §4 Lưu → thấy `Saved adapter → .../adapters/sft-mini` và `Saved merged 16-bit → .../models/sft-merged`.
- [ ] Chạy cell sinh thử (quicksort) → câu trả lời tiếng Việt mạch lạc.
- [ ] Chạy cell giải phóng GPU cuối NB1.
- [ ] **Ghi lại:** số epoch (1), số mẫu (1000), loss đầu/cuối → dùng cho REFLECTION §1.
- [ ] Chạy `backup()` (mục 0.4) + `Ctrl+S`.

---

## 3. NB2 — Dữ liệu sở thích (~2–3 phút, 12 điểm)

- [ ] Chạy cell import + tải tokenizer.
- [ ] §1 Nạp, lọc, chia → thấy `train=800  eval=100  (no prompt overlap)` (assert không trùng phải qua).
- [ ] **Đọc kỹ 3 cặp mẫu** in ra (rubric yêu cầu). Nếu chỉ in 1 cặp, thêm cell:
  ```python
  for r in list(train_ds)[:3]:
      print("PROMPT:", r["prompt"][0]["content"][:300])
      print("CHOSEN:", r["chosen"][0]["content"][:300])
      print("REJECTED:", r["rejected"][0]["content"][:300]); print("-"*80)
  ```
  - [ ] Tự hỏi: `chosen` tốt hơn thật hay chỉ dài hơn? Ghi 1–2 câu nhận xét vào cell văn bản.
- [ ] §2 Thiên vị độ dài → **ghi lại con số** `chosen longer in XX.X% of pairs` và median token chosen/rejected.
- [ ] Ảnh `02b-pref-length.png` được lưu.
- [ ] §3 Lưu → `data/pref/train.parquet`, `data/pref/eval.parquet`, `stats.json`.
- [ ] Giải phóng GPU + chạy `backup()` (mục 0.4) + `Ctrl+S`.

---

## 4. NB3 — Huấn luyện DPO (~40–60 phút, 24 điểm)

- [ ] Cell import: 2 assert `SFT_MERGED` và `train.parquet` phải qua (nếu không → chạy lại NB1/NB2).
- [ ] §1 Nạp `models/sft-merged` + LoRA mới.
- [ ] §2 Huấn luyện:
  - [ ] Dòng in cấu hình: ghi lại `loss_type`, `beta` (0.1), `lr` (5e-6), `max_length`.
  - [ ] Vài phút đầu **không có thanh tiến trình** (đang precompute log-prob reference) → bình thường, đừng dừng.
  - [ ] Chờ ~100 bước, eval held-out mỗi 25 bước.
  - [ ] Ghi lại `train loss` và `held-out reward accuracy` in cuối.
  - [ ] **Ghi lại thời gian chạy** và VRAM cao nhất (thêm cell `print(torch.cuda.max_memory_allocated()/1e9, "GB")`).
- [ ] §3 Vẽ đường reward → ảnh `03-dpo-reward-curves.png` có **chosen và rejected tách riêng, cả train và held-out**.
- [ ] `first logged loss` ≈ 0.693 (nếu lệch xa → reference sai, xem lại NB1).
- [ ] Cell chẩn đoán → **ghi lại nhãn**: `INTENDED` / `LIKELIHOOD DISPLACEMENT` / `FAILURE` / `AMBIGUOUS` + câu giải thích.
- [ ] Tự đọc biểu đồ và ghi chú nhanh (cho REFLECTION §3):
  - [ ] `chosen` train: tăng hay giảm? held-out: tăng hay giảm?
  - [ ] `rejected` train/held-out: giảm nhanh cỡ nào?
  - [ ] Margin tăng vì chosen ↑ hay vì rejected ↓ nhanh hơn?
  - [ ] Held-out có đi cùng hướng với train không (hay chỉ train tăng → học thuộc)?
- [ ] §4 Lưu → `adapters/dpo/` có `adapter_config.json`, `dpo_metrics.json`, `split.json`.
- [ ] Mở `adapters/dpo/dpo_metrics.json` (cell `!cat adapters/dpo/dpo_metrics.json`) → chép số liệu cho REFLECTION §2.
- [ ] Giải phóng GPU + chạy `backup()` (mục 0.4) + `Ctrl+S`.

> Nếu hết giờ/chạy quá lâu: thêm `os.environ["PREF_TRAIN"] = "200"` vào cell cài đặt và chạy lại từ đầu (chỉ để thử, kết quả sẽ yếu hơn).

---

## 5. NB4 — So sánh SFT vs SFT+DPO + chấm tự động (~20–30 phút, 16 điểm)

- [ ] Cell import: assert `split_mismatch` phải qua.
- [ ] §1 Sinh câu trả lời cho 8 câu cố định + ≥ 50 câu held-out → ghi lại `mean chars SFT ... DPO ...`.
- [ ] §2 Bảng 8 câu → ảnh `04-side-by-side-table.png`. Đọc phần in ra, **chọn sẵn 2 ví dụ** cho REFLECTION §4:
  - [ ] 1 câu helpfulness (h1–h4)
  - [ ] 1 câu safety (s1–s4)
- [ ] §3 Chấm bằng hội đồng 2 reward model (tự tải, không cần key) → ghi lại sanity của từng RM (cần ≥ 80%).
- [ ] §4 Tổng hợp → `data/eval/judge_summary.json`, `data/eval/side_by_side.jsonl`.
- [ ] Mở `!cat data/eval/judge_summary.json` và chép:
  - [ ] `heldout`: n, DPO thắng, SFT thắng, hoà, win rate + **CI 95%**
  - [ ] `helpfulness`, `safety`: như trên
  - [ ] `sanity_accuracy`
  - [ ] `longer_answer_won_frac`, `length_matched_win_rate`, `score_length_spearman`
  - [ ] `per_judge` (win rate từng RM: Qwen3 vs Llama) và `judge_agreement`
- [ ] Trả lời nhanh (nháp cho §4):
  - [ ] CI có chứa 0.5 không? (chứa → "chưa đủ bằng chứng", vẫn hợp lệ, viết thật)
  - [ ] DPO thắng vì tốt hơn hay vì dài hơn? (so với tỉ lệ chosen dài hơn ở NB2)
  - [ ] RM Qwen3 có cho DPO thắng cao hơn hẳn RM Llama không? → bàn về preference leakage (cùng họ Skywork/Qwen với mô hình gán nhãn/sinh dữ liệu).
- [ ] Giải phóng GPU + chạy `backup()` (mục 0.4) + `Ctrl+S`.

**Phần bắt buộc chạy xong.** Bonus (mục 9) có thể làm tiếp trong cùng phiên nếu còn GPU.

---

## 6. Đưa kết quả về máy (TRƯỚC KHI ngắt Colab server)

**Đừng ngắt server** cho đến khi xong mục 8 (`verify.py` cần `models/sft-merged` trên server).

- [ ] Chạy `backup()` lần cuối → thấy `✓ Đã push`.
  (File bonus `variants_summary.json`, `grpo_metrics.json` đã có sẵn trong `FILES`.)
- [ ] `Ctrl+S` notebook `colab/Lab22_DPO_T4_executed.ipynb` → output đã nằm trong file trên máy, không cần tải về.
- [ ] Trên máy: `git pull` → kéo `submission/`, `data/`, `adapters/*.json` về.
- [ ] Kiểm tra có đủ 4 ảnh bắt buộc trong `submission/screenshots/`:
  - [ ] `02-sft-loss.png`
  - [ ] `02b-pref-length.png`
  - [ ] `03-dpo-reward-curves.png`
  - [ ] `04-side-by-side-table.png`
- [ ] **Không** tải/commit `models/`, file `.safetensors`, `gguf*/`.

---

## 7. Viết `submission/REFLECTION.md` (20 điểm + 8 điểm giải thích chẩn đoán)

Dùng **số thật** từ file, không ước lượng bằng mắt. Xoá hết placeholder `_<...>_` và `_Trả lời ở đây._` ở các mục bắt buộc (`make verify` sẽ bắt lỗi nếu còn).

- [ ] **Header:** Tên `Nguyễn Hải Nam`, Khoá, Tier `T4`, Ngày.
- [ ] **§1 Cấu hình:** GPU (Colab T4 16 GB), mô hình gốc (`unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit`), SFT (saillab · 1000 · 1 epoch), dữ liệu sở thích (sailor2 vi · 800/100), % chosen dài hơn (NB2), β/lr/epoch, giám khảo + sanity, chi phí (0 đồng).
- [ ] **§2 Kết quả DPO:** thời gian NB3, VRAM, `end_reward_gap`, `eval_reward_accuracy`, margin held-out, `diagnosis`, độ dài trung bình SFT → DPO.
- [ ] **§3 Đọc đường reward (≥ 100 từ):**
  - [ ] Mô tả riêng `chosen` và `rejected`, trên **train và held-out**.
  - [ ] Margin tăng vì chosen ↑ hay rejected ↓ nhanh hơn (likelihood displacement)?
  - [ ] Held-out có cùng hướng train không?
  - [ ] Chẩn đoán tự động có khớp với điều bạn thấy không? Giải thích bằng ý ở NB0 §5.
  - [ ] (Thêm) câu trả lời NB0 về likelihood displacement + thiên vị độ dài.
- [ ] **§4 So sánh SFT vs SFT+DPO:**
  - [ ] Điền bảng 3 dòng (held-out / hữu ích / an toàn) từ `judge_summary.json`.
  - [ ] Điền dòng giám khảo, sanity, `score_length_spearman`.
  - [ ] Bàn: CI có chứa 0.5? giám khảo đáng tin trên tiếng Việt? thắng vì tốt hay vì dài? `per_judge` Qwen3 vs Llama, preference leakage.
  - [ ] 2 ví dụ cụ thể (1 helpfulness, 1 safety) có trích câu trả lời SFT và DPO.
- [ ] **§5 β-sweep:** nếu không chạy bonus → viết **giả thuyết 3 câu** (β lớn giữ gần reference hơn, margin/accuracy thay đổi thế nào).
- [ ] **§6 Một quyết định quan trọng (≥ 150 từ):** chọn 1 (vd β=0.1, lr=5e-6 thay vì 5e-7, giám khảo hội đồng RM thay vì API, 800 cặp…), trả lời đủ 4 ý: phương án thay thế · vì sao chọn · kết quả xác nhận/bất ngờ · làm lại đổi gì.
- [ ] §7–§9: chỉ điền nếu làm bonus tương ứng; tick danh sách bonus ở cuối file.
- [ ] Đếm lại số từ §3 (≥ 100) và §6 (≥ 150).

---

## 8. Commit, kiểm tra, nộp bài

- [ ] `git status` → kiểm tra **không** có `.env`, `models/`, `.safetensors`.
- [ ] Commit các file:
  - [ ] `colab/Lab22_DPO_T4_executed.ipynb` (còn output)
  - [ ] `submission/screenshots/*.png`
  - [ ] `submission/REFLECTION.md`
  - [ ] `data/eval/*.json`, `data/eval/side_by_side.jsonl`
  - [ ] `data/pref/train.parquet`, `data/pref/eval.parquet`
  - [ ] `adapters/dpo/{adapter_config,dpo_metrics,split}.json`, `adapters/sft-mini/adapter_config.json`
  ```bash
  git pull                      # kéo các file server đã push trước
  git add colab submission
  git commit -m "Lab 22: notebook đã chạy + reflection"
  git push
  ```
- [ ] **`make verify` (5 điểm):** script kiểm tra cả `models/sft-merged/` (chỉ có trên server) và đường dẫn
      reference trong adapter (`/content/lab22/models/sft-merged`), nên chỉ qua được **trên chính Colab server đã
      train**. Sau khi push REFLECTION từ máy, chạy cell này trong notebook (server vẫn phải còn):
  ```python
  !git -C /content/repo pull -q
  !cp -rn /content/repo/. /content/lab22/      # -n: không ghi đè kết quả đã chạy, chỉ thêm scripts/, notebooks/, REFLECTION
  !cd /content/lab22 && python scripts/verify.py
  ```
  - [ ] Kết thúc bằng `✓ Core checks passed` → `Ctrl+S` (output nằm trong notebook) → commit + push notebook từ máy.
  - [ ] Nếu chỉ còn lỗi `UNEDITED ... REFLECTION` → sửa REFLECTION trên máy, push, chạy lại cell trên.
  - [ ] Chạy `verify.py` trên máy sẽ báo thiếu `models/sft-merged`: bình thường, vì trọng số không commit.
- [ ] Mở repo trên GitHub (chế độ ẩn danh) → xác nhận repo **public**, ảnh hiển thị, notebook có output.
- [ ] Nộp link repo vào **LMS** trước **23:59 ngày hôm sau** (muộn trừ 10%/ngày, quá 3 ngày = 0).
- [ ] Giữ repo public đến khi có điểm.

---

## 9. Bonus (tuỳ chọn, tối đa +20) — chọn theo thời gian còn lại

Gợi ý thứ tự (điểm/thời gian tốt nhất trước). Chi tiết: [`BONUS-CHALLENGE.md`](BONUS-CHALLENGE.md).

- [ ] **NB3b — biến thể loss (+8):** chạy phần NB3b → ảnh `03b-variants.png`, `adapters/variants/variants_summary.json`. Điền REFLECTION §8: biến thể nào đổi độ dài nhiều nhất và vì sao (dựa vào công thức loss).
- [ ] **NB5 — GGUF (+4):** chạy NB5 (biên dịch llama.cpp 3–5 phút) → `data/eval/deploy_meta.json`; **chụp tay** cell llama-cpp (tên file Q4_K_M + câu trả lời) → `06-gguf-smoke.png`.
- [ ] **NB6 — benchmark (+6):** IFEval / GSM8K / Global-MMLU-vi → `07-benchmark-comparison.png`, `data/eval/benchmark_results.json`; REFLECTION §7 (≥ 150 từ, so Δ với ~2× stderr).
- [ ] **NB7 — GRPO (+8):** → `08-grpo-reward.png`, `adapters/grpo/grpo_metrics.json`; REFLECTION §9 (độ chính xác trước/sau + sai số chuẩn √(p(1−p)/n)).
- [ ] **β-sweep (+6):** cần thư mục `scripts/` trên server (chạy lệnh `cp -rn` ở mục 8) rồi `!make beta-sweep` → `bonus-beta-sweep.png`; REFLECTION §5 (≥ 100 từ).
- [ ] **Chấm chéo (+4):** đặt `JUDGE_PROVIDER` + `JUDGE_MODEL` + key bằng `getpass` như token GitHub ở 0.4 (không dán key vào notebook), chạy lại NB4 §3–§4 → báo `cross_judge.agreement`.
- [ ] **HF Hub (+3):** đẩy `adapters/dpo` + model card (mô hình gốc, dữ liệu, siêu tham số, kết quả đánh giá).

---

## Gặp lỗi?

| Triệu chứng | Xử lý |
|---|---|
| `CUDA out of memory` | Thêm `os.environ["MAX_LEN"] = "512"` vào cell cài đặt → **Restart** kernel → chạy lại từ đầu. |
| `No CUDA GPU` / assert GPU | Kiểm tra lại bước 0.2 (T4 GPU). |
| Mất kết nối server giữa chừng | Nối lại kernel (nếu server cũ còn thì file vẫn còn) → chạy lại phần A + cell token ở 0.4 → chạy tiếp từ NB bị lỗi (mỗi NB tự nạp lại từ đĩa). Nếu đĩa đã bị xoá → phải chạy lại từ NB1. |
| NB3 rất lâu | Bình thường 40–60 phút. Thử nhanh: `PREF_TRAIN=200`. |
| `my_dpo_loss` sai | So với "Đáp số tham chiếu"; kiểm tra dấu và β. |
| Lỗi khác | [`docs/reference.md`](docs/reference.md) mục "Lỗi thường gặp". |
