import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useAgentSocket } from "./useAgentSocket.js";

const USERS = [
  { id: "alice", role: "admin", blurb: "Everything, including access control and data purges" },
  { id: "bob", role: "operator", blurb: "Read metrics, manage alert rules and annotations" },
  { id: "carol", role: "viewer", blurb: "Read-only analytics" },
  { id: "dave", role: "viewer + key manager", blurb: "Read-only, plus API key management" },
];

const SUGGESTIONS = {
  alice: ["Who has which role on this workspace?", "Delete raw events older than 30 days."],
  bob: ["Did the contract_upload error rate change in the last 20 minutes?",
        "Create an alert when contract_upload errors exceed 5% over 15 minutes."],
  carol: ["Which event type has the highest error rate in the last 20 minutes?",
          "Create an alert for api_call latency above 800 ms."],
  dave: ["Create an API key called ci-pipeline.", "List our API keys."],
};

function store(key, value) {
  try {
    if (value === null) window.localStorage.removeItem(key);
    else window.localStorage.setItem(key, value);
  } catch { /* storage unavailable: keep working without it */ }
}
function load(key) {
  try { return window.localStorage.getItem(key); } catch { return null; }
}

const newId = () => (crypto.randomUUID ? crypto.randomUUID() : String(Math.random())).replace(/[^a-z0-9]/gi, "").slice(0, 16);

async function api(path, token, opts = {}) {
  const r = await fetch(path, { ...opts, headers: { "Content-Type": "application/json",
    ...(token ? { Authorization: `Bearer ${token}` } : {}), ...(opts.headers || {}) } });
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
}

function Login({ onLogin }) {
  const [error, setError] = useState(null);
  const pick = async (user) => {
    try {
      const r = await api("/api/login", null, { method: "POST", body: JSON.stringify({ user }) });
      onLogin(r.user, r.token);
    } catch (e) { setError(String(e)); }
  };
  return (
    <div className="login">
      <h1>Agent Console</h1>
      <p className="muted">Sign in as a demo user. Each one has a different role in SpiceDB, so each sees a different set of tools.</p>
      <div className="users">
        {USERS.map((u) => (
          <button key={u.id} className="user-card" onClick={() => pick(u.id)}>
            <span className="user-name">{u.id}</span>
            <span className="badge">{u.role}</span>
            <span className="muted small">{u.blurb}</span>
          </button>
        ))}
      </div>
      {error && <p className="error">{error}</p>}
    </div>
  );
}

function ToolCard({ item }) {
  const [openResult, setOpenResult] = useState(false);
  let args = item.arguments;
  try { args = JSON.stringify(JSON.parse(item.arguments || "{}")); } catch { /* show raw */ }
  return (
    <div className={`tool tool-${item.state}`}>
      <div className="tool-head" onClick={() => setOpenResult((v) => !v)}>
        <span className="tool-dot" />
        <code className="tool-name">{item.name}</code>
        <code className="tool-args">{args}</code>
        <span className="muted small">{item.state === "running" ? "running" : item.ms != null ? `${item.ms} ms` : item.state}</span>
      </div>
      {openResult && item.result && <pre className="tool-result">{item.result}</pre>}
    </div>
  );
}

function Approval({ approval, onDecide }) {
  const [choice, setChoice] = useState(() => Object.fromEntries(approval.calls.map((c) => [c.id, true])));
  return (
    <div className="approval">
      <div className="approval-title">Approval needed</div>
      <p className="muted small">The agent wants to run a destructive action. The run is paused and checkpointed; it waits here even if the server restarts.</p>
      {approval.calls.map((c) => (
        <label key={c.id} className="approval-row">
          <input type="checkbox" checked={choice[c.id]} onChange={(e) => setChoice({ ...choice, [c.id]: e.target.checked })} />
          <code>{c.name}</code> <code className="tool-args">{c.arguments}</code>
        </label>
      ))}
      <div className="row">
        <button className="primary" onClick={() => onDecide(choice)}>Submit decision</button>
        <button onClick={() => onDecide(Object.fromEntries(approval.calls.map((c) => [c.id, false])))}>Decline all</button>
      </div>
    </div>
  );
}

function Chat({ sid, session, user, connected, onAsk, onApprove, onResume, onCancel }) {
  const [text, setText] = useState("");
  const end = useRef(null);
  useEffect(() => { if (end.current) end.current.scrollIntoView({ block: "end" }); }, [session.items.length, session.items[session.items.length - 1]?.text]);
  const busy = session.running || !!session.approval || session.resumable || !connected;
  const submit = (e) => {
    e.preventDefault();
    const t = text.trim();
    if (!t || busy) return;
    onAsk(sid, t);
    setText("");
  };
  return (
    <section className="chat">
      <div className="messages">
        {session.items.length === 0 && (
          <div className="empty">
            <p className="muted">Try:</p>
            {(SUGGESTIONS[user] || []).map((s) => (
              <button key={s} className="suggestion" disabled={!connected} onClick={() => onAsk(sid, s)}>{s}</button>
            ))}
          </div>
        )}
        {session.items.map((it, i) => {
          if (it.kind === "user") return <div key={i} className="msg msg-user">{it.text}</div>;
          if (it.kind === "assistant") return <div key={i} className={`msg msg-assistant${it.open ? " streaming" : ""}`}>{it.text}</div>;
          return <ToolCard key={i} item={it} />;
        })}
        {session.approval && <Approval key={session.approval.calls.map((c) => c.id).join()} approval={session.approval}
                                       onDecide={(d) => onApprove(sid, d)} />}
        {session.resumable && (
          <div className="notice">
            This run stopped before finishing (the server restarted or it was cancelled). Its state is checkpointed.
            <button className="primary" onClick={() => onResume(sid)}>Resume from checkpoint</button>
          </div>
        )}
        {session.error && <div className="notice error">{session.error}</div>}
        <div ref={end} />
      </div>
      <div className="statusbar small muted">
        {session.running ? `${session.stage || "starting"}${session.iteration ? ` · step ${session.iteration}/10` : ""}`
          : session.status === "capped" ? "stopped at the 10-step limit"
          : session.meta ? `done in ${session.meta.duration_ms} ms · ${session.meta.iterations} step(s)` : ""}
        {session.running && <button className="link" onClick={() => onCancel(sid)}>cancel</button>}
      </div>
      <form className="composer" onSubmit={submit}>
        <input value={text} onChange={(e) => setText(e.target.value)} disabled={busy}
               placeholder={!connected ? "Connecting..." : busy ? "Waiting for the agent..." : "Ask about metrics, alerts, keys, access..."} />
        <button className="primary" disabled={busy || !text.trim()}>Send</button>
      </form>
    </section>
  );
}

export default function App() {
  const [user, setUser] = useState(() => load("agent.user"));
  const [token, setToken] = useState(() => load("agent.token"));
  const [me, setMe] = useState(null);
  const [threads, setThreads] = useState([]);
  const [active, setActive] = useState(null);

  const logout = useCallback(() => {
    store("agent.user", null);
    store("agent.token", null);
    setUser(null);
    setToken(null);
    setMe(null);
    setThreads([]);
    setActive(null);
  }, []);
  const { sessions, connected, open, ask, approve, resume, cancel } = useAgentSocket(token, logout);

  useEffect(() => {
    if (!token) return;
    api("/api/me", token).then(setMe).catch(() => logout());
    api("/api/threads", token).then((ts) => {
      setThreads(ts);
      if (ts.length) setActive((a) => a || ts[0].thread_id);
    }).catch(() => {});
  }, [token, logout]);

  useEffect(() => { if (active && connected) open(active); }, [active, connected, open]);

  const titles = useMemo(() => Object.fromEntries(threads.map((t) => [t.thread_id, t.title])), [threads]);
  const sessionIds = useMemo(() => {
    const ids = threads.map((t) => t.thread_id);
    for (const sid of Object.keys(sessions)) if (!ids.includes(sid)) ids.unshift(sid);
    return ids;
  }, [threads, sessions]);

  if (!token) return <Login onLogin={(u, t) => { store("agent.user", u); store("agent.token", t); setUser(u); setToken(t); }} />;

  const startSession = () => setActive(newId());
  const label = (sid) => {
    const first = sessions[sid]?.items.find((i) => i.kind === "user");
    return titles[sid] || first?.text || "New session";
  };
  const dot = (sid) => {
    const s = sessions[sid];
    if (!s) return "";
    if (s.approval) return "dot-approval";
    if (s.running) return "dot-running";
    if (s.resumable) return "dot-resumable";
    return "";
  };

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="me">
          <div><strong>{user}</strong> <span className="badge">{USERS.find((u) => u.id === user)?.role}</span></div>
          <button className="link" onClick={logout}>switch user</button>
        </div>
        {me && (
          <details className="tools-visible">
            <summary>{me.tools.length} tools visible to you</summary>
            <ul>{me.tools.map((t) => <li key={t}><code>{t}</code>{me.needs_approval.includes(t) && <span className="muted small"> · needs approval</span>}</li>)}</ul>
          </details>
        )}
        <button className="primary new" onClick={startSession}>New session</button>
        <nav className="sessions">
          {sessionIds.map((sid) => (
            <button key={sid} className={`session${sid === active ? " active" : ""}`} onClick={() => setActive(sid)}>
              <span className={`dot ${dot(sid)}`} />
              <span className="session-title">{label(sid)}</span>
            </button>
          ))}
        </nav>
        <div className={`conn small ${connected ? "ok" : "bad"}`}>{connected ? "connected" : "reconnecting..."}</div>
      </aside>
      <main>
        {active ? (
          <Chat sid={active} session={sessions[active] || { items: [] }} user={user} connected={connected}
                onAsk={ask} onApprove={approve} onResume={resume} onCancel={cancel} />
        ) : (
          <div className="placeholder muted">Start a session. Several can run at once over the same connection.</div>
        )}
      </main>
    </div>
  );
}
