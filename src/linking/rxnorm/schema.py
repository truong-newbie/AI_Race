"""
RxNorm Knowledge Base Schema

Chua thong tin cac thuoc trong competition data.
Gioi han chi cac RxCUI xuat hien trong rxnorm_linking_samples.jsonl.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RxNormEntry:
    """Mot entry trong RxNorm knowledge base (competition subset)."""
    rxcui: str
    name: str
    name_short: str
    generic_name: str
    ingredient: Optional[str] = None
    strength_value: Optional[float] = None
    strength_unit: Optional[str] = None
    dose_form: Optional[str] = None
    brand: Optional[str] = None
    synonyms: list[str] = field(default_factory=list)
    tty: Optional[str] = None
    category: Optional[str] = None

    def __post_init__(self):
        if self.ingredient is None:
            self.ingredient = self.generic_name
        if self.dose_form is None:
            self._infer_dose_form()

    def _infer_dose_form(self) -> None:
        if "Tablet" in self.name or "tablet" in self.name.lower():
            self.dose_form = "tablet"
        elif "Capsule" in self.name or "capsule" in self.name.lower():
            self.dose_form = "capsule"
        elif "Injection" in self.name or "injection" in self.name.lower():
            self.dose_form = "injection"
        elif "Solution" in self.name or "solution" in self.name.lower():
            self.dose_form = "solution"
        elif "Cream" in self.name or "cream" in self.name.lower():
            self.dose_form = "cream"

    def _parse_strength(self) -> None:
        """Parse strength_value and strength_unit from name if not set."""
        if self.strength_value is not None:
            return
        import re
        m = re.search(r'(\d+(?:[.,]\d+)?)\s*(MG|G|MCG|ML|IU|%)', self.name, re.IGNORECASE)
        if m:
            self.strength_value = float(m.group(1).replace(',', '.'))
            self.strength_unit = m.group(2).upper()

    def get_all_searchable_texts(self) -> list[str]:
        """Tat ca texts co the dung de tim kiem."""
        texts = [self.name, self.name_short, self.generic_name]
        if self.ingredient and self.ingredient != self.generic_name:
            texts.append(self.ingredient)
        texts.extend(self.synonyms)
        if self.brand:
            texts.append(self.brand)
        return [t for t in texts if t]

    def is_combination(self) -> bool:
        return False

    def to_dict(self) -> dict:
        return {
            "rxcui": self.rxcui,
            "name": self.name,
            "name_short": self.name_short,
            "generic_name": self.generic_name,
            "ingredient": self.ingredient,
            "strength_value": self.strength_value,
            "strength_unit": self.strength_unit,
            "dose_form": self.dose_form,
            "brand": self.brand,
            "synonyms": self.synonyms,
            "tty": self.tty,
            "category": self.category,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RxNormEntry":
        return cls(
            rxcui=data["rxcui"],
            name=data.get("name", ""),
            name_short=data.get("name_short", data.get("name", "")),
            generic_name=data.get("generic_name", data.get("ingredient", "")),
            ingredient=data.get("ingredient"),
            strength_value=data.get("strength_value"),
            strength_unit=data.get("strength_unit"),
            dose_form=data.get("dose_form"),
            brand=data.get("brand"),
            synonyms=data.get("synonyms", []),
            tty=data.get("tty"),
            category=data.get("category"),
        )


@dataclass
class ParsedDrug:
    """Ket qua parse mot drug mention."""
    original: str
    ingredients: list[str] = field(default_factory=list)
    strength_values: list[float] = field(default_factory=list)
    strength_units: list[str] = field(default_factory=list)
    dose_form: Optional[str] = None
    brand: Optional[str] = None

    def has_strength(self) -> bool:
        return len(self.strength_values) > 0

    def is_combination(self) -> bool:
        return len(self.ingredients) > 1

    def main_ingredient(self) -> Optional[str]:
        return self.ingredients[0] if self.ingredients else None

    def main_strength(self) -> tuple[Optional[float], Optional[str]]:
        if self.strength_values:
            return self.strength_values[0], self.strength_units[0] if self.strength_units else None
        return None, None


# ---------------------------------------------------------------------------
# Knowledge Base - competition drugs only
# ---------------------------------------------------------------------------

def _make_entry(rxcui: str, name: str, name_short: str, generic_name: str,
                strength: str, brand_names: list[str], category: str) -> RxNormEntry:
    """Helper to create RxNormEntry with parsed strength."""
    import re
    entry = RxNormEntry(
        rxcui=rxcui,
        name=name,
        name_short=name_short,
        generic_name=generic_name,
        ingredient=generic_name,
        dose_form="tablet",
        category=category,
        synonyms=brand_names,
    )
    m = re.search(r'(\d+(?:[.,]\d+)?)\s*(MG|G|MCG|ML|IU|%)', strength, re.IGNORECASE)
    if m:
        entry.strength_value = float(m.group(1).replace(',', '.'))
        entry.strength_unit = m.group(2).upper()
    entry.brand = brand_names[0] if brand_names else None
    return entry


def get_knowledge_base() -> list[RxNormEntry]:
    """
    Lay RxNorm KB tu competition data.

    Chua 28 unique RxCUI tu rxnorm_linking_samples.jsonl.
    """
    return [
        # Analgesics
        _make_entry("161", "Acetaminophen 500 MG Oral Tablet", "Paracetamol 500mg",
                    "Acetaminophen", "500mg", ["Panadol", "Efferalgan", "Tylenol"], "analgesic"),
        _make_entry("1191", "Aspirin 81 MG Oral Tablet", "Aspirin 81mg",
                    "Aspirin", "81mg", ["Bayer", "Bufferin"], "analgesic"),
        _make_entry("1192", "Aspirin 325 MG Oral Tablet", "Aspirin 325mg",
                    "Aspirin", "325mg", ["Bayer"], "analgesic"),

        # Antibiotics
        _make_entry("8628", "Ceftriaxone 1 GM Injection", "Ceftriaxone 1g",
                    "Ceftriaxone", "1g", ["Rocephin"], "antibiotic"),
        _make_entry("723", "Amoxicillin 500 MG Oral Capsule", "Amoxicillin 500mg",
                    "Amoxicillin", "500mg", ["Amoxil", "Moxatag"], "antibiotic"),
        _make_entry("2556", "Ciprofloxacin 500 MG Oral Tablet", "Ciprofloxacin 500mg",
                    "Ciprofloxacin", "500mg", ["Cipro"], "antibiotic"),
        _make_entry("18631", "Azithromycin 500 MG Oral Tablet", "Azithromycin 500mg",
                    "Azithromycin", "500mg", ["Zithromax", "Azithrocin"], "antibiotic"),
        _make_entry("83367", "Cefuroxime 500 MG Oral Tablet", "Cefuroxime 500mg",
                    "Cefuroxime", "500mg", ["Ceftin", "Zinnat"], "antibiotic"),
        _make_entry("1116635", "Metronidazole 500 MG Oral Tablet", "Metronidazole 500mg",
                    "Metronidazole", "500mg", ["Flagyl"], "antibiotic"),

        # Antidiabetics
        _make_entry("6809", "Metformin 500 MG Oral Tablet", "Metformin 500mg",
                    "Metformin", "500mg", ["Glucophage", "Fortamet"], "antidiabetic"),
        _make_entry("860975", "Metformin 850 MG Oral Tablet", "Metformin 850mg",
                    "Metformin", "850mg", ["Glucophage"], "antidiabetic"),
        _make_entry("316672", "Glibenclamide 5 MG Oral Tablet", "Glibenclamide 5mg",
                    "Glibenclamide", "5mg", ["Glyburide", "DiaBeta"], "antidiabetic"),
        _make_entry("861007", "Metformin 1000 MG Oral Tablet", "Metformin 1000mg",
                    "Metformin", "1000mg", ["Glucophage"], "antidiabetic"),
        _make_entry("317541", "Sitagliptin 100 MG Oral Tablet", "Sitagliptin 100mg",
                    "Sitagliptin", "100mg", ["Januvia"], "antidiabetic"),

        # Cardiovascular / GI
        _make_entry("7646", "Omeprazole 20 MG Delayed Release Oral Tablet", "Omeprazole 20mg",
                    "Omeprazole", "20mg", ["Prilosec", "Losec"], "gastrointestinal"),
        _make_entry("32937", "Amlodipine 5 MG Oral Tablet", "Amlodipine 5mg",
                    "Amlodipine", "5mg", ["Norvasc"], "cardiovascular"),
        _make_entry("52175", "Losartan 50 MG Oral Tablet", "Losartan 50mg",
                    "Losartan", "50mg", ["Cozaar"], "cardiovascular"),
        _make_entry("617312", "Atorvastatin 20 MG Oral Tablet", "Atorvastatin 20mg",
                    "Atorvastatin", "20mg", ["Lipitor"], "cardiovascular"),
        _make_entry("197361", "Pantoprazole 40 MG Delayed Release Oral Tablet", "Pantoprazole 40mg",
                    "Pantoprazole", "40mg", ["Protonix"], "gastrointestinal"),
        _make_entry("314076", "Bisoprolol 5 MG Oral Tablet", "Bisoprolol 5mg",
                    "Bisoprolol", "5mg", ["Zebeta"], "cardiovascular"),
        _make_entry("198211", "Spironolactone 25 MG Oral Tablet", "Spironolactone 25mg",
                    "Spironolactone", "25mg", ["Aldactone"], "cardiovascular"),
        _make_entry("855332", "Clopidogrel 75 MG Oral Tablet", "Clopidogrel 75mg",
                    "Clopidogrel", "75mg", ["Plavix"], "cardiovascular"),

        # Steroids
        _make_entry("8640", "Prednisolone 5 MG Oral Tablet", "Prednisolone 5mg",
                    "Prednisolone", "5mg", ["Prelone", "Orapred"], "steroid"),
        _make_entry("285070", "Dexamethasone 4 MG Oral Tablet", "Dexamethasone 4mg",
                    "Dexamethasone", "4mg", ["Decadron"], "steroid"),

        # Psychiatric
        _make_entry("72407", "Sertraline 50 MG Oral Tablet", "Sertraline 50mg",
                    "Sertraline", "50mg", ["Zoloft"], "psychiatric"),
        _make_entry("72509", "Alprazolam 0.5 MG Oral Tablet", "Alprazolam 0.5mg",
                    "Alprazolam", "0.5mg", ["Xanax"], "psychiatric"),
        _make_entry("312961", "Diazepam 5 MG Oral Tablet", "Diazepam 5mg",
                    "Diazepam", "5mg", ["Valium"], "psychiatric"),
        _make_entry("206977", "Zopiclone 7.5 MG Oral Tablet", "Zopiclone 7.5mg",
                    "Zopiclone", "7.5mg", ["Imovane"], "psychiatric"),
    ]
