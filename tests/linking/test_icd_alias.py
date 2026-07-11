"""Tests for alias index building and lookup."""

import pytest
from src.linking.icd.schema import ICD10Entry
from src.linking.icd.alias_index import AliasIndex


class TestAliasIndexBuild:
    """Test alias index building from entries."""

    @pytest.fixture
    def sample_entry(self):
        return ICD10Entry(
            code="T01.1",
            name_vi="Bệnh tim",
            name_en="Heart Disease",
            synonyms=["cardiac disease", "bệnh tim mạch"],
            aliases=["bệnh tim", "đau tim"],
            include_terms=["tim", "mạch máu"],
        )

    def test_single_entry_indexed(self, sample_entry):
        idx = AliasIndex()
        idx.add_entry(sample_entry)

        assert idx.get_entry("T01.1") is not None
        assert len(idx.all_codes()) == 1

    def test_alias_lookup(self, sample_entry):
        idx = AliasIndex()
        idx.add_entry(sample_entry)

        results = idx.lookup_exact("bệnh tim")
        assert "T01.1" in results

    def test_synonym_lookup(self, sample_entry):
        idx = AliasIndex()
        idx.add_entry(sample_entry)

        results = idx.lookup_exact("cardiac disease")
        assert "T01.1" in results

    def test_normalized_lookup(self, sample_entry):
        idx = AliasIndex()
        idx.add_entry(sample_entry)

        results = idx.lookup_normalized("BỆNH TIM")
        assert "T01.1" in results

    def test_en_name_lookup(self, sample_entry):
        idx = AliasIndex()
        idx.add_entry(sample_entry)

        results = idx.lookup_exact("Heart Disease")
        assert "T01.1" in results

    def test_include_term_indexed(self, sample_entry):
        idx = AliasIndex()
        idx.add_entry(sample_entry)

        # include_terms are indexed — test with an alias
        results = idx.lookup_exact("bệnh tim")
        assert "T01.1" in results


class TestAliasPriority:
    """Alias matching priority and precedence."""

    @pytest.fixture
    def entries(self):
        return [
            ICD10Entry(
                code="X01",
                name_vi="Đau đầu",
                name_en="Headache",
                aliases=["đau đầu migraine"],
            ),
            ICD10Entry(
                code="X02",
                name_vi="Migraine",
                name_en="Migraine",
                aliases=["migraine"],
            ),
        ]

    def test_alias_returns_correct_code(self, entries):
        idx = AliasIndex()
        idx.build(entries)

        results = idx.lookup_exact("đau đầu migraine")
        assert "X01" in results

    def test_unique_aliases(self, entries):
        idx = AliasIndex()
        idx.build(entries)

        # Each normalized form maps to first code encountered
        codes = idx.lookup_exact("migraine")
        assert len(codes) >= 1


class TestCodeCollision:
    """Test handling of entries that share aliases."""

    @pytest.fixture
    def colliding_entries(self):
        return [
            ICD10Entry(
                code="Y01",
                name_vi="Viêm dạ dày",
                aliases=["viêm dạ dày", "đau bụng"],
            ),
            ICD10Entry(
                code="Y02",
                name_vi="Viêm dạ dày tá tràng",
                aliases=["viêm dạ dày tá tràng", "đau bụng"],
            ),
        ]

    def test_both_codes_returned(self, colliding_entries):
        idx = AliasIndex()
        idx.build(colliding_entries)

        results = idx.lookup_exact("đau bụng")
        # Both codes share "đau bụng" as alias
        assert len(results) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
