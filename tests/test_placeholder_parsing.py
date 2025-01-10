from app.placeholder_parsing import extract_placeholders


def test_extract_placeholders_default_style():
    """Extract placeholders with default style [placeholders]."""
    text = "Hello [name], welcome to [place]!"
    expected = ["name", "place"]
    result = extract_placeholders(text)
    assert set(result) == set(expected), f"Expected {expected} but got {result}"


def test_extract_placeholders_custom_style():
    """Extract placeholders with a custom style using curly braces {placeholders}."""
    text = "Your order {order_id} is ready for {customer_name}."
    placeholder_style = r"\{(.*?)\}"
    expected = ["order_id", "customer_name"]
    result = extract_placeholders(text, placeholder_style)
    assert set(result) == set(expected), f"Expected {expected} but got {result}"


def test_extract_placeholders_no_placeholders():
    """Handle text with no placeholders."""
    text = "No placeholders here!"
    expected = []
    result = extract_placeholders(text)
    assert set(result) == set(expected), f"Expected {expected} but got {result}"


def test_extract_repeated_placeholders():
    """Handle nested placeholders with default style."""
    text = "Some [placeholder] might do [something] at multiple places. These [placeholder]s are tricky."
    expected = ["placeholder", "something"]
    result = extract_placeholders(text)
    assert set(result) == set(expected), f"Expected {expected} but got {result}"


def test_extract_placeholders_empty_placeholders():
    """Handle empty placeholders."""
    text = "Empty placeholders [][] are here."
    expected = [""]
    result = extract_placeholders(text)
    assert set(result) == set(expected), f"Expected {expected} but got {result}"
