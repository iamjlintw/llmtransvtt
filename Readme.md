# VTT 字幕處理工具

這是一個用於處理 VTT 字幕檔案的 Python 工具。

## 安裝說明

1. 確保您的系統已安裝 Python 3.6 或更高版本
2. 下載此專案到本地目錄
3. 在專案目錄中開啟終端機，執行以下指令安裝所需套件：
bash
pip install -r requirements.txt

## 使用方法

bash
python runvtt.py <輸入檔案名稱> <輸出檔案名稱>

例如：
bash
python runvtt.py input.vtt output.vtt

 

## 注意事項

- 輸入檔案必須是 .vtt 格式的字幕檔
- 輸出檔案會自動建立在同一目錄下
- 如果輸出檔案已存在，將會被覆蓋

## 功能說明

- 自動處理 VTT 字幕檔案的格式
- 修正時間軸標記
- 優化字幕顯示效果


 