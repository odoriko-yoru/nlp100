"""Model of MeCab token with the unidic_lite."""

from dataclasses import dataclass
from typing import List


# ['表層','発音','読み','原型','品詞','活用型','活用形','アクセント型']
@dataclass
class MeCabToken:
    """MeCabの形態素解析結果の1トークンを表すクラス.

    辞書unidic_liteを利用した.
    """

    token: List[str]  # parse結果
    extra_info: str = "unidc_lite"  # 辞書特有の追加情報（UniDicなど）
    original_line: str = ""  # トークンの元の文字列（デバッグ用）

    @property
    def surface(self) -> str:
        """表層形."""
        return self.token[0]

    @property
    def pos(self) -> str:
        """品詞."""
        return self.token[4]

    @property
    def pronunciation(self) -> str:
        """発音."""
        return self.token[1]

    @property
    def reading(self) -> str:
        """読み."""
        return self.token[2]

    @property
    def base(self) -> str:
        """原型."""
        return self.token[3]

    @property
    def conjugation_type(self) -> str:
        """活用形."""
        return self.token[5]

    @property
    def conjugation_form(self) -> str:
        """活用形."""
        return self.token[6]

    def split_pos(self) -> List[str]:
        """Split pos to subtype."""
        if not self.pos:
            return []
        return self.pos.split("-")
