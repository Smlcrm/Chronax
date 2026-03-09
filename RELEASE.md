# Releasing chronax to PyPI

## Prerequisites

1. **Python 3.11+** and a clean virtual environment (recommended).
2. **PyPI account** — [Register](https://pypi.org/account/register/) if needed.
3. **API token** — Create a token at [PyPI → Account settings → API tokens](https://pypi.org/manage/account/token/). Use a project-scoped token for `chronax` (or account token for multiple projects). Keep the token secret (e.g. in a password manager or env var).

## One-time setup

```bash
# From the repo root
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade build twine
```

Optional: configure Twine to use your token so you don’t type it every time:

- **Option A — Environment variable (recommended)**  
  Export before uploading (don’t commit this):
  ```bash
  export TWINE_USERNAME=__token__
  export TWINE_PASSWORD=pypi-xxxxxxxx...
  ```

- **Option B — `.pypirc`**  
  Create `~/.pypirc` (don’t commit it):
  ```ini
  [pypi]
  username = __token__
  password = pypi-xxxxxxxx...
  ```

## Release steps

### 1. Bump version (if needed)

Edit `pyproject.toml` and set the new version under `[project]`:

```toml
version = "0.1.0"   # e.g. 0.1.1 for a patch release
```

Commit and tag (optional but recommended):

```bash
git add pyproject.toml
git commit -m "Release v0.1.0"
git tag v0.1.0
```

### 2. Build the package

From the repo root (with the same venv active):

```bash
pip install --upgrade build
python -m build
```

This creates `dist/` with:

- `chronax-0.1.0.tar.gz` (source distribution)
- `chronax-0.1.0-py3-none-any.whl` (wheel)

### 3. (Recommended) Upload to TestPyPI first

```bash
pip install --upgrade twine
twine upload --repository testpypi dist/*
```

When prompted, use:

- **Username:** `__token__`
- **Password:** your TestPyPI API token from [test.pypi.org/manage/account/token/](https://test.pypi.org/manage/account/token/)

Install and smoke-test:

```bash
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ chronax
python -c "from chronax.models import AutoARIMA; print('OK')"
```

### 4. Upload to PyPI

When you’re satisfied with TestPyPI:

```bash
twine upload dist/*
```

Use your **production** PyPI API token (username `__token__`, password = token).

### 5. Verify on PyPI

- Project page: `https://pypi.org/project/chronax/`
- Install: `pip install chronax`

## Checklist

- [ ] Version bumped in `pyproject.toml`
- [ ] `python -m build` runs without errors
- [ ] (Optional) Tag created: `git tag v0.1.0`
- [ ] TestPyPI upload and install test done
- [ ] PyPI upload completed
- [ ] `pip install chronax` works and version is correct

## Troubleshooting

| Issue | What to do |
|-------|------------|
| `InvalidDistribution` | Delete `dist/` and run `python -m build` again. |
| `File already exists` | You can’t re-upload the same version. Bump version and rebuild. |
| 403 on upload | Check token has correct scope (project `chronax` or account-wide). |
| Import errors after install | Ensure no stray `chronax` in `PYTHONPATH`; use a fresh venv. |

## Notes

- **Never commit** `.pypirc`, `TWINE_PASSWORD`, or any PyPI token.
- `dist/` is in `.gitignore`; don’t commit built artifacts.
- For CI/CD, use a trusted secret for the PyPI token and run `build` + `twine upload` in the release job.
