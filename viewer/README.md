# snek2 viewer

A tiny web viewer for archived games stored as `ArchiveTurnRow` Parquet shards (written by the self-play executor).

## Backend

Runs a Go HTTP server that:
- Scans `-data-dir` for `.parquet` files (skips `tmp/`)
- Uses DuckDB to query the shards
- Exposes JSON APIs:
  - `GET /api/games?limit=&offset=`
  - `GET /api/games/{gameId}/turns`

Run:

```bash
cd /home/brensch/snek2
go run ./viewer -data-dir data/generated -listen 127.0.0.1:8080
```

## Frontend (Vite + React + TypeScript)

Run dev server (proxies `/api` to the backend at `127.0.0.1:8080`):

```bash
cd /home/brensch/snek2/viewer/web
npm install
npm run dev
```

Then open the URL Vite prints (usually `http://localhost:5173`).

## Production build

Build the SPA:

```bash
cd /home/brensch/snek2/viewer/web
npm install
npm run build
```

Serve it from the Go backend:

```bash
cd /home/brensch/snek2
go run ./viewer -data-dir data/generated -listen 127.0.0.1:8080 -static-dir viewer/web/dist
```
