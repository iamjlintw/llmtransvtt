# VTT 字幕翻譯處理工具

這是一個用於英翻中的處理 VTT 字幕檔案的 Python 工具。基於DeepSeek Chat 的API 開發 

## 安裝說明

1. 確保您的系統已安裝 Python 3.6 或更高版本
2. 下載此專案到本地目錄
3. 在專案目錄中開啟終端機，執行以下指令安裝所需套件：
bash
pip install -r requirements.txt

## 使用方法

bash
python runvtt.py <輸入檔案名稱>  

例如：
bash
python runvtt.py input.vtt

 

## 注意事項

- 輸入檔案必須是 .vtt 格式的字幕檔
- 輸入檔案必須放在 `source` 資料夾中
- 翻譯結果會自動存放在 `output` 資料夾中
- 如果輸出檔案已存在，將會被覆蓋
- 程式會自動控制 API 請求頻率，符合 API 供應商的限制（每分鐘最多 10 次請求）


## 功能說明

- 自動處理 VTT 字幕檔案的格式
- 英文轉中文翻譯
- 保留原文，製作雙語字幕
- 智慧批次處理，避免超出 API 限制
- 自動錯誤處理和重試機制
- 進度顯示和詳細日誌輸出

 