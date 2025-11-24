3D-Conway's Game of Life — Commands (中文說明)

說明
- 在 Windows + WSL 的環境下，以下命令請在 PowerShell / cmd 使用；每個需要在 WSL 執行的命令我都加上 `wsl -d Ubuntu -- bash -lc` 前綴以保證到正確環境。

**建置與工具**
- **Build all:** 建置整個專案

```powershell
wsl -d Ubuntu -- bash -lc 'cd /mnt/d/0-IOC-1/PP/final/3D-Conways-Game-of-Life-0.0.1 && make -j4'
```

- **Build viewer/tools only:** 編譯 `tools/visualize_3d_sdl`（SDL/OpenGL viewer）

```powershell
wsl -d Ubuntu -- bash -lc 'cd /mnt/d/0-IOC-1/PP/final/3D-Conways-Game-of-Life-0.0.1 && make tools'
```

**執行模擬（2D / 3D 範例）**
- **2D 模擬（範例）：**

```powershell
wsl -d Ubuntu -- bash -lc 'cd /mnt/d/0-IOC-1/PP/final/3D-Conways-Game-of-Life-0.0.1 && ./main 2d --width 128 --height 128 --iterations 200 --backend cpu --output_every 1'
```

- **3D 模擬（最小範例，會在 `frames_3d/` 寫入切片 PPM）：**

```powershell
wsl -d Ubuntu -- bash -lc 'cd /mnt/d/0-IOC-1/PP/final/3D-Conways-Game-of-Life-0.0.1 && ./main 3d --width 16 --height 16 --depth 4 --iterations 20 --density 0.12 --backend cpu --output_every 1'
```

- **常用 `main` 選項解釋（可帶入上面命令）：**
  - `--width` / `--height` / `--depth` : 空間大小
  - `--iterations` : 要執行多少代
  - `--density` : 初始隨機活細胞密度（0..1）
  - `--backend` : `cpu` / `openmp` / `avx2` / `cuda`（目前非 CPU 的後端會 fallback 到 CPU 參考實作，會印 warning）
  - `--output_every N` : 每 N 代輸出一次（切片輸出到 `frames/` 與 `frames_3d/`）
  - `--seed` : 可選的隨機種子

**檢視（Viewer）**
- **檢視測試模式（內建 synthetic stacks，確認 viewer 運作）：**

```powershell
wsl -d Ubuntu -- bash -lc 'cd /mnt/d/0-IOC-1/PP/final/3D-Conways-Game-of-Life-0.0.1 && ./tools/visualize_3d_sdl --test'
```
輸出會包含像 `Loaded stack series: 4x4x4 (2 iterations)` 的摘要。

- **檢視指定的 `frames_3d/` 目錄（載入該目錄所有可辨識的 iterations/slices）：**

```powershell
wsl -d Ubuntu -- bash -lc 'cd /mnt/d/0-IOC-1/PP/final/3D-Conways-Game-of-Life-0.0.1 && ./tools/visualize_3d_sdl --dir frames_3d --iter 0'
```

- **viewer 常見按鍵 / 操作（內建於 `tools/visualize_3d_sdl`）：**
  - **W/A/S/D** : 前後左右移動（相對於鏡頭）
  - **Space / X** : 向上/向下移動
  - **滑鼠拖曳（按住左鍵）** : 旋轉視角（mouse capture 可用 Tab 切換）
  - **Tab** : 切換滑鼠捕捉（capture）模式
  - **--test** : 顯示 50 顆測試方塊與 3D 格線，便於確認相機與渲染

**檔案 / 產出**
- **3D 切片輸出位置（範例）：** `frames_3d/frame_0000_slice_000.ppm`、`frames_3d/frame_0000_slice_001.ppm` …
- **PPM 格式**: 使用 P6（二進位 RGB）格式； viewer 會掃描 `frames_3d/` 來組合 iterations 與 slices。

**進階 / 開發註記**
- 若要重新做一組乾淨的測試，先清空或移動 `frames_3d/` 再跑模擬，否則 viewer 會載入目錄內全部可辨識的 iterations。

```powershell
wsl -d Ubuntu -- bash -lc 'cd /mnt/d/0-IOC-1/PP/final/3D-Conways-Game-of-Life-0.0.1 && rm -rf frames_3d && mkdir frames_3d'
```

- 若要只編譯單一目標，可直接用 `make <target>`（查看 `Makefile` 內可用的 target，例如 `tools`、`all` 等）。

**範例工作流程（在兩個視窗裡執行）**
1. 在 frame 輸出 terminal（模擬）：
```powershell
wsl -d Ubuntu -- bash -lc 'cd /mnt/d/0-IOC-1/PP/final/3D-Conways-Game-of-Life-0.0.1 && ./main 3d --width 48 --height 48 --depth 12 --iterations 500 --density 0.12 --backend cpu --output_every 1'
```
2. 在 viewer terminal（可即時手動重啟或使用未來自動重載功能）：
```powershell
wsl -d Ubuntu -- bash -lc 'cd /mnt/d/0-IOC-1/PP/final/3D-Conways-Game-of-Life-0.0.1 && ./tools/visualize_3d_sdl --dir frames_3d --iter 0'
```


---
註：如果你想，我可以：
- 實作 viewer 的自動重載 (`--auto` 或檔案監聽)；或
- 把 `--birth` / `--survive` 規則參數化成 CLI 選項；或
- 嘗試實作 OpenMP / AVX2 / CUDA 後端中的一個（需要額外時間與測試）。

檔案位置： `command.md`（專案根目錄）
