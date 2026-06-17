from blue_doc_source.source.document.enrichment import BasicEnricher


def test_extracts_keywords():
    enricher = BasicEnricher(top_k_keywords=3)
    result = enricher.enrich("the cat sat on the mat the cat")
    assert "cat" in result["semantic_hints"]["keywords"]
    assert "mat" in result["semantic_hints"]["keywords"]


def test_extracts_email_entity():
    enricher = BasicEnricher()
    result = enricher.enrich("Contact us at support@example.com for help.")
    emails = [e for e in result["entities"] if e["type"] == "EMAIL"]
    assert any(e["text"] == "support@example.com" for e in emails)


def test_extracts_date_entity():
    enricher = BasicEnricher()
    result = enricher.enrich("The deadline is January 15, 2024.")
    dates = [e for e in result["entities"] if e["type"] == "DATE"]
    assert len(dates) > 0


def test_extracts_percent_entity():
    enricher = BasicEnricher()
    result = enricher.enrich("Revenue grew by 23% last quarter.")
    percents = [e for e in result["entities"] if e["type"] == "PERCENT"]
    assert any("23" in e["text"] for e in percents)


def test_extracts_proper_noun():
    enricher = BasicEnricher()
    result = enricher.enrich("John Smith signed the agreement.")
    proper_nouns = [e for e in result["entities"] if e["type"] == "PROPER_NOUN"]
    assert any("John Smith" in e["text"] for e in proper_nouns)


def test_semantic_hints_word_count():
    enricher = BasicEnricher()
    result = enricher.enrich("one two three four five")
    assert result["semantic_hints"]["word_count"] == 5


def test_semantic_hints_has_questions():
    enricher = BasicEnricher()
    assert enricher.enrich("What is this?")["semantic_hints"]["has_questions"] is True
    assert enricher.enrich("This is a statement.")["semantic_hints"]["has_questions"] is False


def test_semantic_hints_has_numbers():
    enricher = BasicEnricher()
    assert enricher.enrich("There are 5 items.")["semantic_hints"]["has_numbers"] is True
    assert enricher.enrich("No numbers here.")["semantic_hints"]["has_numbers"] is False


def test_empty_text_returns_empty():
    enricher = BasicEnricher()
    result = enricher.enrich("")
    assert result["entities"] == []
    assert result["semantic_hints"]["word_count"] == 0
    assert result["semantic_hints"]["keywords"] == []
