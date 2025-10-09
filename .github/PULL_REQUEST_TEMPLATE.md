# Pull Request Template for Giskard

Thank you for contributing! Please fill out this template to help us review your changes efficiently.  

---

## 🚀 Description

<!-- Provide a clear and concise description of what this PR does. -->

Examples:
- Fix a bug in drift detection
- Improve documentation for ML tests
- Add a new feature in scan module
- Refactor code for better maintainability

---

## 🔗 Related Issue

<!-- Link to the issue this PR addresses (e.g., #2200) -->

---

## 📝 Type of Change

<!-- Mark all that apply with [x] -->

- [ ] 📚 Documentation / examples / tutorials / dependencies update
- [ ] 🔧 Bug fix (non-breaking change)
- [ ] 🥂 Improvement (enhancement of existing functionality)
- [ ] 🚀 New feature (non-breaking)
- [ ] 💥 Breaking change (changes existing functionality)
- [ ] 🔐 Security fix

---

## ✅ Checklist

<!-- Mark all that apply with [x] -->

- [ ] I have read the [`CODE_OF_CONDUCT.md`](https://github.com/Giskard-AI/ai-inspector/blob/master/CODE_OF_CONDUCT.md)
- [ ] I have read the [`CONTRIBUTING.md`](https://github.com/Giskard-AI/ai-inspector/blob/master/CONTRIBUTING.md) guide
- [ ] My code follows the style guide (black, isort, pre-commit hooks)
- [ ] I have added or updated relevant docstrings (Google / NumPy style)
- [ ] I have updated documentation/examples if needed
- [ ] I have added or updated tests for my changes
- [ ] I have verified my code works with supported Python versions
- [ ] I have self-reviewed my code
- [ ] I have checked that this PR is not duplicating an existing issue/PR

---

## 🧪 How to Test / Reproduce

<!-- Explain how reviewers can verify your changes. Skip if the PR is documentation-only. -->

Example:
```python
# Steps to test
from giskard import scan
result = scan.run(my_model, my_dataset)
print(result)
