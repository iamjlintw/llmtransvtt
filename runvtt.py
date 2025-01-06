import re
from typing import List, Dict, Optional, Tuple
import asyncio
from pathlib import Path
import aiohttp
from dataclasses import dataclass
from tqdm import tqdm
import os
from dotenv import load_dotenv

# 載入環境變數
load_dotenv()


@dataclass
class SubtitleBlock:
    index: int
    timestamp: str
    content: str
    translated_content: Optional[str] = None


class WebVTTParser:
    def __init__(self):
        self.blocks = []

    def parse_file(self, file_path: str) -> List[SubtitleBlock]:
        """解析 VTT 檔案"""
        import codecs

        try:
            # 使用 codecs 開啟檔案，自動處理 BOM
            with codecs.open(file_path, 'r', encoding='utf-8-sig') as f:
                lines = [line.strip() for line in f.readlines()]

            # 基本驗證
            if not any(line == 'WEBVTT' for line in lines):
                raise ValueError("找不到 WEBVTT 標記")

            current_block = None
            content_lines = []

            # 遍歷每一行
            for line in lines:
                # 跳過 WEBVTT 和空行
                if not line or line == 'WEBVTT' or line.startswith('NOTE'):
                    continue

                # 檢查時間戳格式
                if '-->' in line and ':' in line:
                    # 如果有前一個區塊，儲存它
                    if current_block and content_lines:
                        current_block.content = '\n'.join(content_lines)
                        self.blocks.append(current_block)
                        content_lines = []

                    # 創建新區塊
                    current_block = SubtitleBlock(
                        index=len(self.blocks) + 1,
                        timestamp=line,
                        content=""
                    )
                # 如果是數字索引，跳過
                elif line.isdigit():
                    continue
                # 否則，這是內容行
                elif current_block is not None:
                    content_lines.append(line)

            # 處理最後一個區塊
            if current_block and content_lines:
                current_block.content = '\n'.join(content_lines)
                self.blocks.append(current_block)

            print(f"\n成功解析 {len(self.blocks)} 個字幕區塊")
            return self.blocks

        except UnicodeError as e:
            print(f"編碼錯誤：{str(e)}")
            # 嘗試其他編碼
            encodings = ['utf-16', 'big5', 'cp950', 'gb18030']
            for enc in encodings:
                try:
                    with codecs.open(file_path, 'r', encoding=enc) as f:
                        lines = [line.strip() for line in f.readlines()]
                    print(f"使用 {enc} 編碼成功")
                    return self.parse_file(file_path)
                except UnicodeError:
                    continue
            raise ValueError(f"無法讀取檔案，嘗試過的編碼：utf-8-sig, {', '.join(encodings)}")

        except Exception as e:
            print(f"解析錯誤：{str(e)}")
            raise


class WebVTTTranslator:
    def __init__(self):
        """初始化翻譯器，從環境變數讀取設定"""
        self.api_key = os.getenv('API_KEY')
        self.api_base = os.getenv('API_BASE', 'https://api.deepseek.com')
        self.model = os.getenv('API_MODEL', 'deepseek-chat')
        self.temperature = float(os.getenv('API_TEMPERATURE', '1.3'))

        if not self.api_key:
            raise ValueError("未設定 API_KEY 環境變數")
            
        # 設定 API 端點 (針對 deepseek API)
        self.endpoint = f"{self.api_base}/v1/chat/completions"

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
            
        self.MAX_TOKENS = 32000

    def calculate_tokens(self, text: str) -> int:
        """使用 utf8_mb4 編碼計算預估的 token 數量"""
        # 使用 utf8_mb4 編碼計算字節數
        byte_count = len(text.encode('utf8'))
        # 一般來說，每 4 個字節約等於 1 個 token
        return int(byte_count / 4) + 1  # +1 作為安全餘量

    def create_translation_batches(self, blocks: List[SubtitleBlock]) -> List[List[SubtitleBlock]]:
        """建立翻譯批次，使用 utf8_mb4 計算 token"""
        batches = []
        current_batch = []
        current_tokens = 0
        
        # 調整為 DeepSeek-V3 的限制
        MAX_INPUT_TOKENS = 4000  # 輸入限制
        SYSTEM_PROMPT_TOKENS = 500
        AVAILABLE_TOKENS = MAX_INPUT_TOKENS - SYSTEM_PROMPT_TOKENS
        
        # 降低每批次的區塊數限制
        MAX_BLOCKS_PER_BATCH = 15  # 更保守的區塊數限制
        
        # 使用更保守的 token 限制
        SOFT_TOKEN_LIMIT = AVAILABLE_TOKENS * 0.5  # 使用 50% 的可用 tokens

        for block in blocks:
            block_tokens = self.calculate_tokens(block.content)
            
            # 使用更嚴格的限制條件
            if (current_tokens + block_tokens > SOFT_TOKEN_LIMIT or 
                len(current_batch) >= MAX_BLOCKS_PER_BATCH):
                if current_batch:
                    print(f"批次大小：{len(current_batch)} 區塊，{current_tokens} tokens")
                    batches.append(current_batch)
                current_batch = [block]
                current_tokens = block_tokens
            else:
                current_batch.append(block)
                current_tokens += block_tokens

        if current_batch:
            print(f"最後批次大小：{len(current_batch)} 區塊，{current_tokens} tokens")
            batches.append(current_batch)

        print(f"將 {len(blocks)} 個字幕區塊分成 {len(batches)} 批處理")
        return batches

    async def translate_batch(self, batch: List[SubtitleBlock], target_lang: str) -> None:
        """翻譯一個批次的字幕"""
        # 使用數字索引作為分隔標記
        texts_with_index = [f"[{i+1}] {block.content}" for i, block in enumerate(batch)]
        combined_text = "\n".join(texts_with_index)

        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": f"""你是專業字幕翻譯員。請將以下編號內容翻譯成{target_lang}。
                            每段文字都有編號標記 [數字]，請在翻譯時保持相同的編號格式。
                            範例輸入：
                            [1] Hello world
                            [2] How are you
                            
                            範例輸出：
                            [1] 你好世界
                            [2] 你好嗎
                            
                            請確保：
                            1. 每個翻譯都保留原始的編號標記
                            2. 翻譯要準確且通順
                            3. 保持原文的段落數量
                            4. 不要增加或減少編號"""
                        },
                        {
                            "role": "user",
                            "content": combined_text
                        }
                    ],
                    "temperature": self.temperature,
                    "max_tokens": 8000  # 為 DeepSeek-V3 指定最大輸出長度
                }

                async with session.post(
                    self.endpoint,
                    headers=self.headers,
                    json=payload
                ) as response:
                    result = await response.json()
                    
                    if response.status != 200:
                        raise ValueError(f"API 錯誤：狀態碼 {response.status}, 回應：{result}")

                    if 'choices' not in result or not result['choices']:
                        raise ValueError("API 未返回翻譯結果")

                    # 解析帶編號的翻譯結果
                    content = result['choices'][0]['message']['content']
                    translations = []
                    
                    # 使用正則表達式匹配帶編號的翻譯
                    pattern = r'\[(\d+)\](.*?)(?=\[\d+\]|$)'
                    matches = re.finditer(pattern, content, re.DOTALL)
                    
                    # 建立翻譯映射
                    trans_dict = {}
                    for match in matches:
                        idx = int(match.group(1))
                        text = match.group(2).strip()
                        trans_dict[idx] = text
                    
                    # 按原始順序重建翻譯列表
                    translations = []
                    for i in range(1, len(batch) + 1):
                        if i in trans_dict:
                            translations.append(trans_dict[i])
                        else:
                            translations.append("【翻譯失敗】")

                    if len(translations) != len(batch):
                        print(f"\n警告：翻譯結果數量({len(translations)})與原文數量({len(batch)})不符")
                        print("正在嘗試修復...")
                        
                        while len(translations) < len(batch):
                            translations.append("【翻譯失敗】")
                        translations = translations[:len(batch)]

                    for block, translation in zip(batch, translations):
                        block.translated_content = translation.strip()

        except Exception as e:
            print(f"\n批次翻譯出錯: {str(e)}")
            for block in batch:
                block.translated_content = "【翻譯失敗】"
            raise

    async def translate_subtitle(self, blocks: List[SubtitleBlock], target_lang: str) -> None:
        """翻譯所有字幕，包含進度顯示和錯誤處理"""
        batches = self.create_translation_batches(blocks)
        total_batches = len(batches)

        print(f"\n總共分成 {total_batches} 個批次進行翻譯")

        with tqdm(total=total_batches, desc="翻譯進度") as pbar:
            for batch in batches:
                try:
                    await self.translate_batch(batch, target_lang)
                    # 添加延遲以避免 API 限制
                    await asyncio.sleep(2)
                except Exception as e:
                    print(f"\n批次翻譯出錯: {str(e)}")
                    continue
                finally:
                    pbar.update(1)

    def save_translated_vtt(self, blocks: List[SubtitleBlock], output_path: str) -> None:
        """保存翻譯後的雙語字幕檔案"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("WEBVTT\n\n")

            for block in blocks:
                f.write(f"{block.timestamp}\n")
                f.write(f"{block.content}\n")
                if block.translated_content:
                    f.write(f"{block.translated_content}\n")
                else:
                    f.write("【翻譯失敗】\n")
                f.write("\n")


async def main():
    import sys

    # 檢查必要的環境變數
    required_env_vars = ['API_KEY', 'API_BASE']
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"錯誤：請在 .env 檔案中設定以下環境變數：{', '.join(missing_vars)}")
        return

    # 檢查環境變數是否已設定
    if not os.getenv('API_KEY'):
        print("錯誤：請在 .env 檔案中設定 API_KEY")
        return

    if len(sys.argv) < 2:
        print("請指定輸入檔案，例如: python script.py input.vtt")
        return

    input_file = sys.argv[1]
    output_file = input_file.rsplit('.', 1)[0] + '_cht.' + input_file.rsplit('.', 1)[1]

    try:
        # 解析字幕
        parser = WebVTTParser()
        print(f"正在解析字幕檔案: {input_file}")
        blocks = parser.parse_file(input_file)
        print(f"共解析出 {len(blocks)} 個字幕區塊")

        # 翻譯字幕
        translator = WebVTTTranslator()
        print("開始翻譯...")
        await translator.translate_subtitle(blocks, target_lang="中文")

        # 保存結果
        print(f"\n正在保存翻譯結果至: {output_file}")
        translator.save_translated_vtt(blocks, output_file)
        print("翻譯完成！")

    except Exception as e:
        print(f"錯誤：{str(e)}")
        return


if __name__ == "__main__":
    asyncio.run(main())