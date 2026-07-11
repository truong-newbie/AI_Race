"""
Variant Generator for Synthetic Medical Data

Tạo các biến thể từ samples:
- Case variations (lowercase, uppercase, title case)
- Diacritics variations
- Typos và spelling errors
- Punctuation variations
- Whitespace normalization
- Text augmentation
"""

import random
import re
import json
import unicodedata
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field

from .schema import Sample, Entity, EntityType


# =============================================================================
# Case Variations
# =============================================================================

def lowercase(text: str) -> str:
    """Convert to lowercase while preserving diacritics."""
    # Vietnamese lowercase conversions
    replacements = {
        'A': 'a', 'B': 'b', 'C': 'c', 'D': 'd', 'E': 'e', 'F': 'f', 'G': 'g',
        'H': 'h', 'I': 'i', 'J': 'j', 'K': 'k', 'L': 'l', 'M': 'm', 'N': 'n',
        'O': 'o', 'P': 'p', 'Q': 'q', 'R': 'r', 'S': 's', 'T': 't', 'U': 'u',
        'V': 'v', 'W': 'w', 'X': 'x', 'Y': 'y', 'Z': 'z',
        'À': 'à', 'Á': 'á', 'Ả': 'ả', 'Ã': 'ã', 'Ạ': 'ạ',
        'È': 'è', 'É': 'é', 'Ẻ': 'ẻ', 'Ẽ': 'ẽ', 'Ẹ': 'ẹ',
        'Ì': 'ì', 'Í': 'í', 'Ỉ': 'ỉ', 'Ĩ': 'õ', 'Ị': 'ị',
        'Ò': 'ò', 'Ó': 'ó', 'Ỏ': 'ỏ', 'Õ': 'õ', 'Ọ': 'ọ',
        'Ù': 'ù', 'Ú': 'ú', 'Ủ': 'ủ', 'Ũ': 'ũ', 'Ụ': 'ụ',
        'Ỳ': 'ỳ', 'Ý': 'ý', 'Ỷ': 'ỷ', 'Ỹ': 'ỹ', 'Ỵ': 'ỵ',
        'Â': 'â', 'Ê': 'ê', 'Ô': 'ô', 'Ơ': 'ơ', 'Ư': 'ư',
        'Ă': 'ă', 'Đ': 'đ', 'Ơ': 'ơ', 'Ư': 'ư',
    }
    result = text
    for old, new in replacements.items():
        result = result.replace(old, new)
    return result


def title_case(text: str) -> str:
    """Convert to title case while preserving diacritics."""
    # Simple title case - capitalize first letter of each word
    words = text.split()
    result = []
    for word in words:
        if not word:
            continue
        # Capitalize first char, lowercase the rest
        result.append(word[0].upper() + lowercase(word[1:]) if len(word) > 1 else word.upper())
    return ' '.join(result)


def uppercase(text: str) -> str:
    """Convert to uppercase."""
    replacements = {
        'a': 'A', 'b': 'B', 'c': 'C', 'd': 'D', 'e': 'E', 'f': 'F', 'g': 'G',
        'h': 'H', 'i': 'I', 'j': 'J', 'k': 'K', 'l': 'L', 'm': 'M', 'n': 'N',
        'o': 'O', 'p': 'P', 'q': 'Q', 'r': 'R', 's': 'S', 't': 'T', 'u': 'U',
        'v': 'V', 'w': 'W', 'x': 'X', 'y': 'Y', 'z': 'Z',
        'à': 'À', 'á': 'Á', 'ả': 'Ả', 'ã': 'Ã', 'ạ': 'Ạ',
        'è': 'È', 'é': 'É', 'ẻ': 'Ẻ', 'ẽ': 'Ẽ', 'ẹ': 'Ẹ',
        'ì': 'Ì', 'í': 'Í', 'ỉ': 'Ỉ', 'ĩ': 'Ĩ', 'ị': 'Ị',
        'ò': 'Ò', 'ó': 'Ó', 'ỏ': 'Ỏ', 'õ': 'Õ', 'ọ': 'Ọ',
        'ù': 'Ù', 'ú': 'Ú', 'ủ': 'Ủ', 'ũ': 'Ũ', 'ụ': 'Ụ',
        'ỳ': 'Ỳ', 'ý': 'Ý', 'ỷ': 'Ỷ', 'ỹ': 'Ỹ', 'ỵ': 'Ỵ',
        'â': 'Â', 'ê': 'Ê', 'ô': 'Ô', 'ơ': 'Ơ', 'ư': 'Ư',
        'ă': 'Ă', 'đ': 'Đ', 'ơ': 'Ơ', 'ư': 'Ư',
    }
    result = text
    for old, new in replacements.items():
        result = result.replace(old, new)
    return result


# =============================================================================
# Diacritics Variations
# =============================================================================

# Common Vietnamese diacritic variants
DIACRITIC_VARIANTS = {
    # Acute vowels
    'á': ['à', 'a', 'ả', 'ã', 'ạ'],
    'é': ['è', 'e', 'ẻ', 'ẽ', 'ẹ'],
    'í': ['ì', 'i', 'ỉ', 'ĩ', 'ị'],
    'ó': ['ò', 'o', 'ỏ', 'õ', 'ọ'],
    'ú': ['ù', 'u', 'ủ', 'ũ', 'ụ'],
    'ớ': ['ờ', 'ơ', 'ở', 'ỡ', 'ợ'],
    'ứ': ['ừ', 'ư', 'ử', 'ữ', 'ự'],
    'ý': ['ỳ', 'y', 'ỷ', 'ỹ', 'ỵ'],

    # Breve vowels
    'ă': ['a', 'à', 'á', 'ả', 'ã', 'ạ'],
    'ắ': ['ă', 'à', 'á', 'ả', 'ã', 'ạ'],

    # Hook vowels
    'ơ': ['o', 'ô', 'ò', 'ó', 'ỏ', 'õ', 'ọ'],

    # Tones
    'â': ['a', 'à', 'á', 'ả', 'ã', 'ạ'],
    'ê': ['e', 'è', 'é', 'ẻ', 'ẽ', 'ẹ'],
    'ô': ['o', 'ò', 'ó', 'ỏ', 'õ', 'ọ'],
    'đ': ['d'],
}

# Reverse mapping
REMOVE_DIACRITICS = {
    'à': 'a', 'á': 'a', 'ả': 'a', 'ã': 'a', 'ạ': 'a',
    'è': 'e', 'é': 'e', 'ẻ': 'e', 'ẽ': 'e', 'ẹ': 'e',
    'ì': 'i', 'í': 'i', 'ỉ': 'i', 'ĩ': 'i', 'ị': 'i',
    'ò': 'o', 'ó': 'o', 'ỏ': 'o', 'õ': 'o', 'ọ': 'o',
    'ù': 'u', 'ú': 'u', 'ủ': 'u', 'ũ': 'u', 'ụ': 'u',
    'ỳ': 'y', 'ý': 'y', 'ỷ': 'y', 'ỹ': 'y', 'ỵ': 'y',
    'ả': 'a', 'ẩ': 'a', 'ẫ': 'a', 'ậ': 'a',
    'ể': 'e', 'ế': 'e', 'ễ': 'e', 'ệ': 'e',
    'ỏ': 'o', 'ổ': 'o', 'ỗ': 'o', 'ộ': 'o',
    'ủ': 'u', 'ứ': 'u', 'ừ': 'u', 'ử': 'u', 'ữ': 'u', 'ự': 'u',
    'ở': 'o', 'ợ': 'o', 'ờ': 'o', 'ỡ': 'o',
    'ă': 'a', 'ắ': 'a', 'ằ': 'a', 'ặ': 'a', 'ẳ': 'a', 'ẵ': 'a',
    'â': 'a', 'ấ': 'a', 'ầ': 'a', 'ậ': 'a', 'ẩ': 'a', 'ẫ': 'a',
    'ê': 'e', 'ế': 'e', 'ề': 'e', 'ệ': 'e', 'ể': 'e', 'ễ': 'e',
    'ô': 'o', 'ố': 'o', 'ồ': 'o', 'ộ': 'o', 'ổ': 'o', 'ỗ': 'o',
    'ơ': 'o', 'ớ': 'o', 'ờ': 'o', 'ợ': 'o', 'ở': 'o', 'ỡ': 'o',
    'ư': 'u', 'ứ': 'u', 'ừ': 'u', 'ự': 'u', 'ử': 'u', 'ữ': 'u',
    'đ': 'd',
}


def remove_diacritics(text: str) -> str:
    """Remove all diacritics from text."""
    result = []
    for char in text:
        result.append(REMOVE_DIACRITICS.get(char, char))
    return ''.join(result)


def random_diacritic_error(text: str, error_rate: float = 0.1) -> str:
    """
    Randomly replace some diacritics with variants.

    Args:
        text: Input text
        error_rate: Probability of changing each diacritic character

    Returns:
        Text with diacritic variations
    """
    result = []
    for char in text:
        if char in DIACRITIC_VARIANTS and random.random() < error_rate:
            variants = DIACRITIC_VARIANTS[char]
            # Pick a different variant
            other_variants = [v for v in variants if v != char]
            if other_variants:
                result.append(random.choice(other_variants))
            else:
                result.append(char)
        else:
            result.append(char)
    return ''.join(result)


def normalize_diacritics(text: str) -> str:
    """
    Normalize diacritics to NFD form and remove combining marks.
    """
    # NFC normalization
    normalized = unicodedata.normalize('NFC', text)
    return normalized


# =============================================================================
# Typos and Spelling Errors
# =============================================================================

# Common typos and misspellings
TYPO_MAP = {
    # Common letter swaps
    'ă': 'â',
    'â': 'ă',
    'ơ': 'ô',
    'ư': 'ơ',
    # Common misspellings
    'viêm phổi': ['viêm phổi', 'vịêm phổi', 'viêm phổii'],
    'đái tháo đường': ['đái tháo đường', 'đái tháo đường', 'đại tháo đường'],
    'tăng huyết áp': ['tăng huyết áp', 'tăng huyết áp', 'tăng huyết áp'],
    'paracetamol': ['paracetamol', 'parasetamol', 'panadol'],
    'ceftriaxone': ['ceftriaxone', 'ceftriaxon'],
    'omeprazole': ['omeprazole', 'omeprazol'],
}

# Keyboard proximity errors (Qwerty-like)
KEYBOARD_PROXIMITY = {
    'a': ['q', 's', 'z'],
    'b': ['v', 'n', 'g', 'h'],
    'c': ['x', 'v', 'd', 'f'],
    'd': ['s', 'e', 'r', 'f', 'c', 'x'],
    'e': ['w', 's', 'd', 'r'],
    'f': ['d', 'r', 't', 'g', 'v', 'c'],
    'g': ['f', 't', 'y', 'h', 'b', 'v'],
    'h': ['g', 't', 'y', 'u', 'j', 'n', 'b'],
    'i': ['u', 'j', 'k', 'o'],
    'j': ['h', 'y', 'u', 'i', 'k', 'm', 'n'],
    'k': ['j', 'u', 'i', 'o', 'l', 'm'],
    'l': ['k', 'i', 'o', 'p'],
    'm': ['n', 'j', 'k'],
    'n': ['b', 'h', 'j', 'm'],
    'o': ['i', 'k', 'l', 'p'],
    'p': ['o', 'l'],
    'q': ['w', 'a'],
    'r': ['e', 'd', 'f', 't'],
    's': ['a', 'w', 'e', 'd', 'x', 'z'],
    't': ['r', 'f', 'g', 'y'],
    'u': ['y', 'j', 'k', 'i'],
    'v': ['c', 'f', 'g', 'b'],
    'w': ['q', 'a', 's', 'e'],
    'x': ['s', 'd', 'c'],
    'y': ['t', 'g', 'h', 'u'],
    'z': ['a', 's', 'x'],
    # Vietnamese specific
    'đ': ['d'],
    'ă': ['â', 'a'],
    'ô': ['o', 'ơ'],
    'ơ': ['ô', 'ư'],
    'ư': ['ơ'],
}


def keyboard_typo(text: str, num_errors: int = 1) -> str:
    """
    Generate typo by pressing adjacent keys.

    Args:
        text: Input text
        num_errors: Number of typos to introduce

    Returns:
        Text with keyboard-induced typos
    """
    chars = list(text)
    chars_lower = [c.lower() for c in chars]

    # Find valid positions (alphabetic characters)
    valid_positions = [i for i, c in enumerate(chars_lower) if c in KEYBOARD_PROXIMITY]
    if not valid_positions:
        return text

    # Randomly select positions to modify
    positions_to_modify = random.sample(valid_positions, min(num_errors, len(valid_positions)))

    for pos in positions_to_modify:
        char = chars_lower[pos]
        if char in KEYBOARD_PROXIMITY:
            new_char = random.choice(KEYBOARD_PROXIMITY[char])
            # Preserve original case
            if chars[pos].isupper():
                new_char = new_char.upper()
            chars[pos] = new_char

    return ''.join(chars)


def random_char_insertion(text: str) -> str:
    """Insert random character."""
    if not text:
        return text

    pos = random.randint(0, len(text))
    char = random.choice('abcdefghijklmnopqrstuvwxyzăâđêôơư')
    return text[:pos] + char + text[pos:]


def random_char_deletion(text: str) -> str:
    """Delete random character."""
    if len(text) <= 1:
        return text

    pos = random.randint(0, len(text) - 1)
    return text[:pos] + text[pos + 1:]


def random_char_swap(text: str) -> str:
    """Swap two adjacent characters."""
    if len(text) <= 2:
        return text

    pos = random.randint(0, len(text) - 2)
    chars = list(text)
    chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
    return ''.join(chars)


def generate_typo(text: str, typo_type: Optional[str] = None) -> str:
    """
    Generate typo variant of text.

    Args:
        text: Input text
        typo_type: Specific type of typo, or None for random

    Returns:
        Text with typo
    """
    typo_types = ['keyboard', 'insertion', 'deletion', 'swap']

    if typo_type is None:
        typo_type = random.choice(typo_types)

    if typo_type == 'keyboard':
        return keyboard_typo(text, num_errors=1)
    elif typo_type == 'insertion':
        return random_char_insertion(text)
    elif typo_type == 'deletion':
        return random_char_deletion(text)
    elif typo_type == 'swap':
        return random_char_swap(text)
    else:
        return text


# =============================================================================
# Punctuation Variations
# =============================================================================

PUNCTUATION_PATTERNS = [
    (r'\.\s*', '.'),  # Normalize spacing after period
    (r',\s*', ', '),  # Normalize comma
    (r';\s*', '; '),  # Normalize semicolon
    (r':\s*', ': '),  # Normalize colon
    (r'\s+', ' '),  # Normalize whitespace
]


def normalize_punctuation(text: str) -> str:
    """Normalize punctuation and spacing."""
    result = text
    for pattern, replacement in PUNCTUATION_PATTERNS:
        result = re.sub(pattern, replacement, result)
    return result.strip()


def add_extra_punctuation(text: str) -> str:
    """Add extra punctuation marks."""
    result = text
    # Add period if missing
    if not result.rstrip().endswith('.'):
        result = result + '.'
    return result


def remove_optional_punctuation(text: str) -> str:
    """Remove optional punctuation like extra spaces, periods."""
    result = text
    result = re.sub(r'\s+', ' ', result)
    result = re.sub(r'\.\s*$', '', result)  # Remove trailing period
    return result.strip()


# =============================================================================
# Whitespace Variations
# =============================================================================

def normalize_whitespace(text: str) -> str:
    """Normalize all whitespace to single spaces."""
    return re.sub(r'\s+', ' ', text).strip()


def add_extra_spaces(text: str) -> str:
    """Add extra spaces between words."""
    words = text.split()
    return '  '.join(words)  # Double space


def remove_spaces(text: str) -> str:
    """Remove all spaces."""
    return re.sub(r'\s+', '', text)


# =============================================================================
# Text Augmentation
# =============================================================================

PATIENT_PRONOUNS = [
    "Bệnh nhân",
    "BN",
    "Người bệnh",
    "Bệnh",
    "BN nam",
    "BN nữ",
]

VERB_FORMS = {
    "ho": ["ho", "ho khan", "ho đờm"],
    "sốt": ["sốt", "sốt cao", "nóng"],
    "đau": ["đau", "đau đớn", "đau nhức"],
}

CLINICAL_VERBS = {
    "diagnosis": ["chẩn đoán", "phát hiện", "xác định"],
    "treatment": ["điều trị", "kê đơn", "cho dùng"],
}


def augment_with_pronoun(text: str) -> str:
    """Add patient pronoun to sentence."""
    pronoun = random.choice(PATIENT_PRONOUNS)
    # Check if text already starts with a pronoun
    first_word = text.split()[0] if text.split() else ""
    if first_word in ["Bệnh", "BN", "Người"]:
        return text
    return f"{pronoun} {text[0].lower()}{text[1:]}"


def paraphrase_simple(text: str) -> str:
    """Simple paraphrasing by synonym replacement."""
    result = text

    # Simple word swaps
    swaps = {
        "bệnh nhân": random.choice(["BN", "người bệnh", "bệnh"]),
        "ho": random.choice(["ho", "ho khan", "ho đờm"]),
        "sốt": random.choice(["sốt", "sốt cao", "nóng sốt"]),
        "đau": random.choice(["đau", "đau nhức", "đau đớn"]),
        "dùng": random.choice(["dùng", "sử dụng", "uống"]),
    }

    for old, new_opts in swaps.items():
        new = random.choice(new_opts) if isinstance(new_opts, list) else new_opts
        if old in result:
            # Case-insensitive replacement
            pattern = re.compile(re.escape(old), re.IGNORECASE)
            result = pattern.sub(new, result)

    return result


# =============================================================================
# Variant Generator
# =============================================================================

@dataclass
class VariantConfig:
    """Configuration for variant generation."""
    case_variations: bool = True
    diacritic_variations: bool = True
    typo_variations: bool = True
    punctuation_variations: bool = True
    whitespace_variations: bool = False
    augmentation: bool = False
    typo_rate: float = 0.1
    max_variants_per_sample: int = 5


class VariantGenerator:
    """
    Generator để tạo các biến thể từ samples.

    Usage:
        generator = VariantGenerator()
        variants = generator.generate_variants(sample)
    """

    def __init__(self, config: Optional[VariantConfig] = None, seed: Optional[int] = None):
        """
        Initialize generator.

        Args:
            config: Configuration for variant generation
            seed: Random seed
        """
        self.config = config or VariantConfig()
        if seed is not None:
            random.seed(seed)
        self.counter = 0

    def _next_id(self, base_id: str) -> str:
        """Generate variant ID."""
        self.counter += 1
        return f"{base_id}_v{self.counter}"

    def _adjust_entity_positions(self, text: str, old_text: str, entities: List[Entity]) -> List[Entity]:
        """
        Adjust entity positions after text modification.

        Since we're only modifying the whole text (case, punctuation, etc.)
        and not the entity text itself, positions remain valid if the
        entity text is preserved.
        """
        # Verify entity texts still exist
        new_entities = []
        for entity in entities:
            # Check if entity text is still in new text
            if entity.text in text:
                # Find new position
                start = text.find(entity.text)
                end = start + len(entity.text)
                new_entities.append(Entity(
                    text=entity.text,
                    start=start,
                    end=end,
                    type=entity.type,
                    assertions=entity.assertions.copy(),
                    candidates=entity.candidates.copy(),
                ))
        return new_entities

    def _case_variant(self, sample: Sample) -> Optional[Sample]:
        """Generate case variation."""
        text = sample.text

        variant_type = random.choice(['lower', 'upper', 'title'])
        if variant_type == 'lower':
            new_text = lowercase(text)
        elif variant_type == 'upper':
            new_text = uppercase(text)
        else:
            new_text = title_case(text)

        # Only create variant if different
        if new_text == text:
            return None

        entities = self._adjust_entity_positions(new_text, text, sample.entities)
        if not entities:
            return None

        return Sample(
            id=self._next_id(sample.id),
            text=new_text,
            entities=entities,
            source="variant_case",
            review_status="auto_validated",
            metadata={"variant_type": variant_type, "original_id": sample.id},
        )

    def _diacritic_variant(self, sample: Sample) -> Optional[Sample]:
        """Generate diacritic variation."""
        text = sample.text

        # Randomly choose between removing diacritics or introducing errors
        variant_type = random.choice(['remove', 'error'])
        if variant_type == 'remove':
            new_text = remove_diacritics(text)
        else:
            new_text = random_diacritic_error(text, error_rate=self.config.typo_rate)

        if new_text == text:
            return None

        entities = self._adjust_entity_positions(new_text, text, sample.entities)
        if not entities:
            return None

        return Sample(
            id=self._next_id(sample.id),
            text=new_text,
            entities=entities,
            source="variant_diacritic",
            review_status="auto_validated",
            metadata={"variant_type": variant_type, "original_id": sample.id},
        )

    def _typo_variant(self, sample: Sample) -> Optional[Sample]:
        """Generate typo variation."""
        text = sample.text

        # Pick one entity text to introduce typo
        if not sample.entities:
            return None

        entity = random.choice(sample.entities)
        original_text = entity.text

        # Generate typo
        typo_text = generate_typo(original_text)

        if typo_text == original_text:
            return None

        # Replace in full text
        new_text = text.replace(original_text, typo_text, 1)

        if new_text == text:
            return None

        # Create new entities
        start = new_text.find(typo_text)
        if start == -1:
            return None
        end = start + len(typo_text)

        new_entities = []
        for e in sample.entities:
            if e.text == original_text:
                new_entities.append(Entity(
                    text=typo_text,
                    start=start,
                    end=end,
                    type=e.type,
                    assertions=e.assertions.copy(),
                    candidates=e.candidates.copy(),
                ))
            else:
                new_entities.append(e)

        return Sample(
            id=self._next_id(sample.id),
            text=new_text,
            entities=new_entities,
            source="variant_typo",
            review_status="auto_validated",
            metadata={"original_id": sample.id},
        )

    def _punctuation_variant(self, sample: Sample) -> Optional[Sample]:
        """Generate punctuation variation."""
        text = sample.text

        variant_type = random.choice(['normalize', 'add', 'remove'])
        if variant_type == 'normalize':
            new_text = normalize_punctuation(text)
        elif variant_type == 'add':
            new_text = add_extra_punctuation(text)
        else:
            new_text = remove_optional_punctuation(text)

        if new_text == text:
            return None

        entities = self._adjust_entity_positions(new_text, text, sample.entities)
        if not entities:
            return None

        return Sample(
            id=self._next_id(sample.id),
            text=new_text,
            entities=entities,
            source="variant_punctuation",
            review_status="auto_validated",
            metadata={"variant_type": variant_type, "original_id": sample.id},
        )

    def _whitespace_variant(self, sample: Sample) -> Optional[Sample]:
        """Generate whitespace variation."""
        text = sample.text

        variant_type = random.choice(['normalize', 'extra', 'remove'])
        if variant_type == 'normalize':
            new_text = normalize_whitespace(text)
        elif variant_type == 'extra':
            new_text = add_extra_spaces(text)
        else:
            new_text = remove_spaces(text)

        if new_text == text:
            return None

        entities = self._adjust_entity_positions(new_text, text, sample.entities)
        if not entities:
            return None

        return Sample(
            id=self._next_id(sample.id),
            text=new_text,
            entities=entities,
            source="variant_whitespace",
            review_status="auto_validated",
            metadata={"variant_type": variant_type, "original_id": sample.id},
        )

    def _augmentation_variant(self, sample: Sample) -> Optional[Sample]:
        """Generate augmentation variant."""
        text = sample.text

        new_text = paraphrase_simple(text)

        if new_text == text:
            return None

        entities = self._adjust_entity_positions(new_text, text, sample.entities)
        if not entities:
            return None

        return Sample(
            id=self._next_id(sample.id),
            text=new_text,
            entities=entities,
            source="variant_augment",
            review_status="auto_validated",
            metadata={"original_id": sample.id},
        )

    def generate_variants(self, sample: Sample) -> List[Sample]:
        """
        Generate all configured variants of a sample.

        Args:
            sample: Input sample

        Returns:
            List of variant samples
        """
        variants = []

        variant_methods = []
        if self.config.case_variations:
            variant_methods.append(self._case_variant)
        if self.config.diacritic_variations:
            variant_methods.append(self._diacritic_variant)
        if self.config.typo_variations:
            variant_methods.append(self._typo_variant)
        if self.config.punctuation_variations:
            variant_methods.append(self._punctuation_variant)
        if self.config.whitespace_variations:
            variant_methods.append(self._whitespace_variant)
        if self.config.augmentation:
            variant_methods.append(self._augmentation_variant)

        # Generate up to max variants
        random.shuffle(variant_methods)
        for method in variant_methods[:self.config.max_variants_per_sample]:
            variant = method(sample)
            if variant:
                variants.append(variant)

        return variants

    def generate_variants_batch(self, samples: List[Sample]) -> List[Sample]:
        """
        Generate variants for multiple samples.

        Args:
            samples: List of input samples

        Returns:
            List of all variants
        """
        all_variants = []
        for sample in samples:
            variants = self.generate_variants(sample)
            all_variants.extend(variants)
        return all_variants


# =============================================================================
# CLI
# =============================================================================

def main():
    """CLI for variant generator."""
    import argparse
    from .schema import load_jsonl, save_jsonl

    parser = argparse.ArgumentParser(description="Generate variants from samples")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output JSONL file")
    parser.add_argument("--max-variants", "-m", type=int, default=5, help="Max variants per sample")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed")
    parser.add_argument("--no-case", action="store_true", help="Disable case variations")
    parser.add_argument("--no-diacritic", action="store_true", help="Disable diacritic variations")
    parser.add_argument("--no-typo", action="store_true", help="Disable typo variations")
    parser.add_argument("--no-punctuation", action="store_true", help="Disable punctuation variations")
    args = parser.parse_args()

    # Load samples
    samples = load_jsonl(args.input)
    samples = [Sample.from_dict(d) for d in samples]

    # Configure generator
    config = VariantConfig(
        case_variations=not args.no_case,
        diacritic_variations=not args.no_diacritic,
        typo_variations=not args.no_typo,
        punctuation_variations=not args.no_punctuation,
        max_variants_per_sample=args.max_variants,
    )

    generator = VariantGenerator(config=config, seed=args.seed)
    variants = generator.generate_variants_batch(samples)

    # Save variants
    save_jsonl(args.output, [v.to_dict() for v in variants])

    print(f"Generated {len(variants)} variants from {len(samples)} samples")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
