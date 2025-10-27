import pytest

from src.clean import basic_clean

assert basic_clean("Hello, WORLD!!!") == "hello world"
assert basic_clean("Visit https://openai.com for info!") == "visit for info"
assert basic_clean("123 Apples & Oranges!!") == "apples oranges"
assert basic_clean("   Extra   spaces   ") == "extra spaces"

print("✅ basic_clean() works correctly")

