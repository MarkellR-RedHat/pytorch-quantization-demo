# Contributing to PyTorch Quantization Demo

Thank you for your interest in contributing to the PyTorch Quantization Demo project!

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/pytorch-quantization-demo.git`
3. Create a feature branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Run tests: `pytest tests/`
6. Commit your changes: `git commit -m "Description of changes"`
7. Push to your fork: `git push origin feature/your-feature-name`
8. Open a Pull Request

## Development Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt
```

## Code Style

- `ruff check app tests scripts` must pass (configuration is in `pyproject.toml`)
- Match the style of the surrounding code
- Any number shown on the dashboard needs a source: the benchmark file, a capture, or a test that checks it

## Testing

All new features should include tests. Run the test suite before submitting:

```bash
pytest --cov=app
```

## Pull Request Process

1. Update the README.md with details of changes if applicable
2. Ensure all tests pass
3. Update documentation as needed
4. Request review from maintainers

## Questions?

Contact: mrawls@redhat.com

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
