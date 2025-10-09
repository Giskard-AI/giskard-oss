try:
    from num2words import num2words
    print("num2words is installed and importable.")
except ImportError as e:
    print("ImportError:", e)
