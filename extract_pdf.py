import fitz

doc = fitz.open(r"test\adwya-etats-financiers-annuels-31-12-2022.pdf")
page = doc[1]  # Page 2 (0-indexed)
text = page.get_text("text")
print("=" * 80)
print("PDF PAGE 2 TEXT CONTENT (First 5000 characters):")
print("=" * 80)
print(text[:5000])
print("\n" + "=" * 80)
print("FULL PAGE LENGTH:", len(text), "characters")
print("=" * 80)
doc.close()
