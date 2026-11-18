"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

const API = (process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/$/, "");
type TripOption = { trip_id: string; partition: string; gps_points: number; candidate_observations: number };
type DecoderResult = { predicted_arc_ids: number[]; predicted_route: number[][]; reference_arc_accuracy: number | null; labeled_observations: number; labeled_reference_arc_missing_from_candidates: number; route_continuity: number | null; continuous_transitions: number; eligible_short_gap_transitions: number; long_gap_steps_relaxed: number; steps_with_no_link_between_candidate_sets: number };
type TripPayload = { trip_id: string; partition: string; is_hidden_evaluator_data: boolean; gps_point_count: number; network_nodes: number; network_arcs: number; candidate_observation_groups: number; reference_route_arc_count: number; track: number[][]; road_candidates: number[][][]; reference_route: number[][]; decoders: { classifier_only: DecoderResult; hmm_viterbi: DecoderResult }; data_note: string };
type View = { x: number; y: number; w: number; h: number };

const pct = (n: number | null) => n == null || !Number.isFinite(n) ? "—" : `${(n * 100).toFixed(1)}%`;
const fmt = (n: number) => n.toLocaleString("en-US");

function projector(data: TripPayload) {
  const all = [...data.track, ...data.reference_route, ...data.decoders.hmm_viterbi.predicted_route, ...data.road_candidates.flat()];
  const lons = all.map(p => p[0]), lats = all.map(p => p[1]);
  const centerLat = (Math.min(...lats) + Math.max(...lats)) / 2;
  const centerLon = (Math.min(...lons) + Math.max(...lons)) / 2;
  const cos = Math.cos(centerLat * Math.PI / 180);
  const project = (p: number[]) => [(p[0] - centerLon) * 111320 * cos, -(p[1] - centerLat) * 111320];
  const xy = all.map(project), minX = Math.min(...xy.map(p => p[0])), maxX = Math.max(...xy.map(p => p[0]));
  const minY = Math.min(...xy.map(p => p[1])), maxY = Math.max(...xy.map(p => p[1]));
  const pad = Math.max(maxX - minX, maxY - minY) * 0.08 + 20;
  const view = { x: minX - pad, y: minY - pad, w: maxX - minX + pad * 2, h: maxY - minY + pad * 2 };
  return { project, view };
}

function MapView({ data, decoder, view, onView }: { data: TripPayload; decoder: "hmm_viterbi" | "classifier_only"; view: View | null; onView: (v: View) => void }) {
  const svg = useRef<SVGSVGElement>(null), drag = useRef<{ x: number; y: number; view: View } | null>(null);
  const { project, view: fitView } = useMemo(() => projector(data), [data]);
  const current = view || fitView;
  const pathD = useCallback((points: number[][]) => points.map((p, i) => { const [x, y] = project(p); return `${i ? "L" : "M"}${x},${y}`; }).join(" "), [project]);
  const pred = data.decoders[decoder].predicted_route;
  const stride = Math.max(1, Math.ceil(data.track.length / 80));
  return <svg ref={svg} className="map" viewBox={`${current.x} ${current.y} ${current.w} ${current.h}`} role="img" aria-label="GPS track, generated road candidates, predicted route and reference route" onWheel={e => { e.preventDefault(); const factor = e.deltaY < 0 ? .88 : 1.14, w = current.w * factor, h = current.h * factor; onView({ x: current.x + (current.w - w) / 2, y: current.y + (current.h - h) / 2, w, h }); }} onPointerDown={e => { drag.current = { x: e.clientX, y: e.clientY, view: current }; e.currentTarget.setPointerCapture(e.pointerId); }} onPointerMove={e => { if (!drag.current || !svg.current) return; const rect = svg.current.getBoundingClientRect(), d = drag.current; onView({ ...d.view, x: d.view.x - (e.clientX - d.x) / rect.width * d.view.w, y: d.view.y - (e.clientY - d.y) / rect.height * d.view.h }); }} onPointerUp={() => { drag.current = null; }}>
    <g>{data.road_candidates.map((line, i) => <path key={`r${i}`} className="road" d={pathD(line)} />)}</g>
    <path className="reference" d={pathD(data.reference_route)} />
    <path className="track" d={pathD(data.track)} />
    <path className="predicted" d={pathD(pred)} />
    {data.track.filter((_, i) => i % stride === 0).map((p, i) => { const [x, y] = project(p); return <circle key={`g${i}`} className="gps-dot" cx={x} cy={y} r={Math.max(current.w, current.h) * .004} />; })}
  </svg>;
}

export default function Home() {
  const [trips, setTrips] = useState<TripOption[]>([]), [tripId, setTripId] = useState(""), [data, setData] = useState<TripPayload | null>(null);
  const [decoder, setDecoder] = useState<"hmm_viterbi" | "classifier_only">("hmm_viterbi");
  const [loadingTrips, setLoadingTrips] = useState(true), [loadingTrip, setLoadingTrip] = useState(false), [error, setError] = useState("");
  const [apiOnline, setApiOnline] = useState(false), [view, setView] = useState<View | null>(null);

  useEffect(() => {
    let active = true;
    (async () => {
      try {
        await fetch(`${API}/api/health`).then(r => { if (!r.ok) throw new Error("API health request failed"); return r.json(); });
        const response = await fetch(`${API}/api/trips`); if (!response.ok) throw new Error(`Trip list request failed (${response.status})`);
        const body = await response.json(); if (!active) return;
        setApiOnline(true); setTrips(body.trips); setTripId(body.trips[0]?.trip_id || "");
      } catch (e) { if (active) setError(`Could not connect to Laptop 1 API at ${API}. Check the API address and Wi-Fi connection.`); }
      finally { if (active) setLoadingTrips(false); }
    })();
    return () => { active = false; };
  }, []);

  useEffect(() => {
    if (!tripId) return;
    let active = true; setLoadingTrip(true); setError("");
    fetch(`${API}/api/trips/${encodeURIComponent(tripId)}`).then(async r => { if (!r.ok) throw new Error(`Trip request failed (${r.status})`); return r.json(); }).then(d => { if (active) { setData(d); setView(null); } }).catch(e => { if (active) setError(e.message); }).finally(() => { if (active) setLoadingTrip(false); });
    return () => { active = false; };
  }, [tripId]);

  const zoom = (factor: number) => { if (!data) return; const base = view || projector(data).view, w = base.w * factor, h = base.h * factor; setView({ x: base.x + (base.w - w) / 2, y: base.y + (base.h - h) / 2, w, h }); };
  const result = data?.decoders[decoder];

  return <main className="shell">
    <header className="top"><div className="brand"><div className="mark">⌁</div><div><h1>PathSnap</h1><p>GPS route matching · evaluator workspace</p></div></div><div className={`status ${apiOnline ? "online" : "offline"}`}><i />{apiOnline ? "Laptop 1 API connected" : loadingTrips ? "Connecting to API" : "API unavailable"}</div></header>
    <section className="intro"><div><h2>Track to road route</h2><p>Inspect model predictions against the supplied reference route.</p></div><div className="test-note">Demo data: 10 supplied <b>student-test</b> trips. No hidden evaluator partition was provided.</div></section>
    {error && <div className="error" role="alert">{error}</div>}
    <div className="layout">
      <section className="card map-card"><div className="toolbar"><div className="select-wrap"><label htmlFor="trip">Choose trip</label><select id="trip" value={tripId} disabled={loadingTrips} onChange={e => setTripId(e.target.value)}>{loadingTrips ? <option>Loading trips…</option> : trips.map(t => <option key={t.trip_id} value={t.trip_id}>Trip {t.trip_id} · {fmt(t.gps_points)} GPS points</option>)}</select></div><div className="tool-right"><select className="decoder" aria-label="Choose decoder" value={decoder} onChange={e => setDecoder(e.target.value as typeof decoder)}><option value="hmm_viterbi">HMM / Viterbi</option><option value="classifier_only">Classifier only</option></select><div className="zoom"><button onClick={() => zoom(.78)} aria-label="Zoom in">+</button><button onClick={() => zoom(1.28)} aria-label="Zoom out">−</button><button onClick={() => setView(null)}>Fit</button></div></div></div>
        <div className="map-wrap">{data && <MapView data={data} decoder={decoder} view={view} onView={setView} />}{loadingTrip && <div className="loading">Requesting trip prediction from Laptop 1…</div>}{!data && loadingTrips && <div className="loading">Connecting to the backend…</div>}<div className="legend"><span><i className="swatch road"/>Candidates</span><span><i className="swatch gps"/>GPS track</span><span><i className="swatch predicted"/>Predicted route</span><span><i className="swatch ref"/>Reference route</span></div><div className="map-hint">Scroll or use buttons to zoom · drag to pan</div></div>
        <div className="map-caption"><span>{data ? `Trip ${data.trip_id} · ${fmt(data.gps_point_count)} GPS points · ${fmt(data.road_candidates.length)} unique generated candidate arcs` : "Waiting for trip data"}</span><span>Local road view · no external map tiles</span></div>
      </section>
      <aside className="side">
        <div className="card panel"><h3>Selected decoder results</h3><div className="metrics"><div className="metric"><strong>{pct(result?.reference_arc_accuracy ?? null)}</strong><span>Reference-arc accuracy</span></div><div className="metric"><strong>{pct(result?.route_continuity ?? null)}</strong><span>Short-gap continuity</span></div><div className="metric"><strong>{result ? `${fmt(result.continuous_transitions)} / ${fmt(result.eligible_short_gap_transitions)}` : "—"}</strong><span>Connected transitions</span></div><div className="metric"><strong>{result ? fmt(result.labeled_observations) : "—"}</strong><span>Labeled observations</span></div></div><p className="footnote">Accuracy counts labeled points whose reference arc is missing from candidates as incorrect. Continuity checks same-arc or directly connected consecutive arcs for gaps ≤ 5 seconds.</p></div>
        <div className="card panel"><h3>Trip details</h3><div className="trip-meta"><div className="meta"><b>{data ? fmt(data.gps_point_count) : "—"}</b><span>GPS points</span></div><div className="meta"><b>{data ? fmt(data.candidate_observation_groups) : "—"}</b><span>Candidate observations</span></div><div className="meta"><b>{data ? fmt(data.network_arcs) : "—"}</b><span>Network arcs</span></div><div className="meta"><b>{data ? fmt(data.reference_route_arc_count) : "—"}</b><span>Reference route arcs</span></div></div><div className="pill">Tuned Hist Gradient Boosting</div></div>
        <div className="card panel"><h3>Decoder comparison · this trip</h3><table className="compare"><thead><tr><th>Decoder</th><th>Ref. accuracy</th><th>Continuity</th></tr></thead><tbody>{data && ([ ["Classifier only", data.decoders.classifier_only, "classifier_only"], ["HMM / Viterbi", data.decoders.hmm_viterbi, "hmm_viterbi"] ] as const).map(([name, row, key]) => <tr key={key} className={decoder === key ? "active-row" : ""}><td>{name}</td><td>{pct(row.reference_arc_accuracy)}</td><td>{pct(row.route_continuity)}</td></tr>)}</tbody></table><p className="footnote">Continuity uses consecutive represented observations with gaps up to five seconds.</p></div>
      </aside>
    </div>
    <div className="footer">PathSnap · Laptop 2 frontend → Laptop 1 FastAPI · selected model family: Hist Gradient Boosting</div>
  </main>;
}
