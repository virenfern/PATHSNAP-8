# PathSnap: two-laptop frontend and API integration

This deliverable follows the required split exactly:

- **Laptop 1 — Backend / ML server:** FastAPI, tuned Hist Gradient Boosting model, candidate features, HMM/Viterbi, trip road-network data, and evaluation logic.
- **Laptop 2 — Frontend / evaluator UI:** Next.js application only. It requests trip IDs and results from Laptop 1 over the local Wi-Fi/hotspot network.

The app lists all 10 supplied `student_test` trips. These are the available demo/evaluation examples; they are not a hidden evaluator partition. The UI displays this distinction.

## Laptop 1 — Backend / ML server

Copy the `backend` folder to Laptop 1. Open PowerShell in that folder and run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --access-log
```

Find Laptop 1's Wi-Fi/hotspot IPv4 address with `ipconfig` (usually in the `192.168.x.x` or `10.x.x.x` range). Keep this terminal open. FastAPI loads the model and test candidate table at startup. Check `http://127.0.0.1:8000/api/health` on Laptop 1.

If Windows Firewall prompts, allow Python on the **Private** network so Laptop 2 can reach port 8000. If the firewall already blocks it, create a private-network inbound TCP rule for port 8000 through Windows Defender Firewall settings. Do not use a public network profile for the demo.

## Laptop 2 — Next.js frontend

Copy the `frontend` folder to Laptop 2. Install Node.js 20.9 or newer. In PowerShell, run:

```powershell
Copy-Item .env.example .env.local
notepad .env.local
```

Replace the example API address with Laptop 1's actual IPv4 address, preserving the port, for example:

```text
NEXT_PUBLIC_API_BASE_URL=http://192.168.1.42:8000
```

The address above is only an example—use the IP shown by `ipconfig` on Laptop 1. Then run:

```powershell
npm install
npm run dev
```

Open `http://localhost:3000` in the Laptop 2 browser. Both laptops must be connected to the same Wi-Fi/hotspot. The Next.js client calls Laptop 1 directly; there is no proxy or second backend on Laptop 2.

## App features

- Ten-trip selector populated by `GET /api/trips`.
- Trip request sent to `GET /api/trips/{trip_id}`.
- Local SVG map draws the actual GPS track, generated candidate arcs, supplied reference route, and selected predicted route. It needs no online map tiles.
- Classifier-only and HMM/Viterbi decoder comparison.
- Per-trip reference-arc accuracy, route continuity, and transition counts.
- Loading state, API connection status, and request error display.
- FastAPI access logs for requests.

## API contract

- `GET /api/health` — status, selected model family, trip count.
- `GET /api/trips` — supplied trip IDs and summary counts.
- `GET /api/trips/{trip_id}` — coordinates, route predictions, reference geometry, and metrics for the selected trip.
- `GET /docs` — interactive FastAPI API documentation.

CORS is configured for localhost/loopback and private LAN IPv4 addresses on port 3000. If the frontend runs from a different origin, set `CORS_ORIGINS` on Laptop 1 to the exact frontend origin(s), comma-separated.

## Metric notes

Reference-arc accuracy uses observations with an assigned reference arc. If the correct arc is not in a point's candidate set, the prediction is counted wrong. Unlabeled observations are excluded. Route continuity counts consecutive represented GPS observations with a gap up to five seconds whose predicted arcs are identical or directly connected in the directed graph. Candidate coverage is about 74.7% across all supplied student-test observations, so candidate generation remains a system limit.

## Folder ownership

- `backend/` belongs on Laptop 1 only; it contains the model and road/data files.
- `frontend/` belongs on Laptop 2; it contains the Next.js UI and the configurable API URL.
