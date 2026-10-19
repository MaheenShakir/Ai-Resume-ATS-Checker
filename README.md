# 📄 AI Resume ATS Checker

Upload a resume (PDF, DOCX or TXT) and get an **ATS score out of 100**, a category breakdown, prioritised improvements, missing keywords, strengths and example rewrites. Optionally paste a job description for a job-specific score.

Built with **Streamlit** and **Google Gemini Flash**.

## Features
- PDF / DOCX / TXT parsing with clear errors (scanned, encrypted or corrupted files)
- Weighted ATS score: Keywords 30%, Formatting 20%, Content 20%, Structure 15%, Impact 15%
- Prioritised fixes (High / Medium / Low), missing keywords, rewrite examples
- Local checks: email, phone, LinkedIn/GitHub, sections, word count
- Downloadable report

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
Get a free API key at <https://aistudio.google.com/apikey> and paste it in the sidebar, or set it once:

```bash
export GEMINI_API_KEY="your-key"   # Windows PowerShell: $env:GEMINI_API_KEY="your-key"
```

Or create `.streamlit/secrets.toml` (already git-ignored):
```toml
GEMINI_API_KEY = "your-key"
```

## Push to GitHub
```bash
git init
git add app.py requirements.txt README.md .gitignore
git commit -m "Add AI Resume ATS Checker"
git branch -M main
# create an empty repo on github.com first (no README), then:
git remote add origin https://github.com/<your-username>/resume-ats-checker.git
git push -u origin main
```
Never commit your API key.

## Deploy on Streamlit Community Cloud (free)
1. Go to <https://share.streamlit.io> and sign in with GitHub.
2. Click **Create app** → pick your repo, branch `main`, main file `app.py`.
3. Open **Advanced settings → Secrets** and paste:
   ```toml
   GEMINI_API_KEY = "your-key"
   ```
4. Click **Deploy**. You get a public URL to put on LinkedIn.

## Troubleshooting
| Problem | Fix |
|---|---|
| "API key rejected" | Re-copy the key from AI Studio; make sure no spaces |
| Quota / rate limit | Wait a minute; free tier has limits |
| "Almost no text extracted" | Resume is a scanned image - export a text-based PDF/DOCX |
| Model not found | Change the model name in the sidebar (default `gemini-flash-latest`) |

## Note
The ATS score is an AI estimate of how well a typical ATS can parse and rank your resume. It is not the score from any specific employer's system.
