---
name: Streamlit preview restart
description: Replit preview behavior while the Streamlit app refreshes match data after a workflow restart.
---

After a workflow restart, the preview may briefly appear blank while the app's synchronous `download_data` call refreshes match data. The preview rendered normally once the data load finished; a restart can also close the browser WebSocket without indicating an application exception.

**Why:** During a reported runtime error, the app returned HTTP 200, the prediction test passed, and the blank preview cleared after the data refresh completed.

**How to apply:** For a blank preview immediately after restarting this app, inspect workflow output and allow the data refresh to finish before changing application code.
