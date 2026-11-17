# Running this project on Replit

This project is a Streamlit football match calculator.

## Start the app

Run the Replit workflow or start it from the Shell with:

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 5000
```

The first visit to a league downloads and caches its match data in `data/cache/`.
An API-Football key is optional; without it, team news can be entered manually.
If configured, store it as the `API_FOOTBALL_KEY` environment secret. Do not
commit API keys to the repository.
