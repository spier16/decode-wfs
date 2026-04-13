# Publishing to PyPI

## First-time setup

1. Install tools: `pip install hatchling build twine`
2. Create a PyPI account at https://pypi.org and enable 2FA
3. Generate an API token: PyPI → Account Settings → API tokens
4. Save credentials so you're not prompted every time:

   ```ini
   # ~/.pypirc
   [pypi]
     username = __token__
     password = pypi-YOUR_TOKEN_HERE
   ```

## Releasing a new version

1. Bump `version` in `pyproject.toml`
2. Commit the change
3. Build: `python -m build`
4. Upload: `twine upload dist/*`
5. Verify: `pip install --upgrade decode-wfs`

> PyPI does not allow overwriting an existing version — every upload needs a new version number.
