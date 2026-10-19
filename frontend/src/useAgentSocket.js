import { useCallback, useEffect, useReducer, useRef, useState } from "react";

// One WebSocket per tab carries every session; frames are routed by their `session` field.

function blank() {
  return { items: [], running: false, stage: null, iteration: 0, approval: null, resumable: false,
           error: null, status: null, meta: null };
}

function parseResult(text) {
  try {
    const v = JSON.parse(text);
    return v && typeof v === "object" && !Array.isArray(v) && v.error ? "error" : "ok";
  } catch {
    return /permission denied|unauthenticated|unknown tool/i.test(text) ? "error" : "ok";
  }
}

function fromTranscript(messages) {
  const items = [];
  for (const m of messages || []) {
    if (m.role === "user") items.push({ kind: "user", text: m.text });
    else if (m.role === "assistant") items.push({ kind: "assistant", text: m.text, open: false });
    else if (m.role === "tool_call") items.push({ kind: "tool", id: m.id, name: m.name, arguments: m.arguments, state: "pending" });
    else if (m.role === "tool_result") {
      const t = items.find((i) => i.kind === "tool" && i.id === m.id);
      if (t) Object.assign(t, { result: m.result, state: parseResult(m.result) });
    }
  }
  return items;
}

function closeAssistant(items) {
  const last = items[items.length - 1];
  if (last && last.kind === "assistant" && last.open) return [...items.slice(0, -1), { ...last, open: false }];
  return items;
}

function applyFrame(s, f) {
  switch (f.type) {
    case "state":
      return { ...s, items: fromTranscript(f.messages), running: f.running, approval: f.approval,
               resumable: f.resumable, status: f.status, error: null };
    case "stage":
      return { ...s, running: true, resumable: false, error: null, stage: f.stage, iteration: f.iteration || s.iteration,
               items: f.stage === "agent" ? closeAssistant(s.items) : s.items };
    case "token": {
      const last = s.items[s.items.length - 1];
      if (last && last.kind === "assistant" && last.open)
        return { ...s, items: [...s.items.slice(0, -1), { ...last, text: last.text + f.text }] };
      return { ...s, items: [...s.items, { kind: "assistant", text: f.text, open: true }] };
    }
    case "tool_start": {
      // A call restored from the transcript (e.g. one waiting for approval) is updated in place.
      if (s.items.some((i) => i.kind === "tool" && i.id === f.id))
        return { ...s, items: s.items.map((i) => (i.kind === "tool" && i.id === f.id ? { ...i, state: "running" } : i)) };
      return { ...s, items: [...closeAssistant(s.items),
        { kind: "tool", id: f.id, name: f.name, arguments: f.arguments, state: "running" }] };
    }
    case "tool_end":
      return { ...s, items: s.items.map((i) => (i.kind === "tool" && i.id === f.id
        ? { ...i, state: f.ok ? "ok" : "error", result: f.result, ms: f.ms } : i)) };
    case "approval_required":
      return { ...s, running: false, stage: "approval", approval: { calls: f.calls } };
    case "done":
      return { ...s, running: false, stage: null, status: f.status, items: closeAssistant(s.items),
               meta: { iterations: f.iterations, duration_ms: f.duration_ms } };
    case "error":
      return { ...s, running: false, stage: null, error: f.message };
    case "cancelled":
      return { ...s, running: false, stage: null, items: closeAssistant(s.items), resumable: true };
    default:
      return s;
  }
}

function reducer(state, action) {
  const sid = action.sid || (action.frame && action.frame.session);
  if (!sid) return state;
  const cur = state[sid] || blank();
  switch (action.type) {
    case "ensure":
      return state[sid] ? state : { ...state, [sid]: blank() };
    case "user":
      return { ...state, [sid]: { ...cur, error: null, running: true, resumable: false,
        items: [...cur.items, { kind: "user", text: action.text }] } };
    case "approval_sent":
      return { ...state, [sid]: { ...cur, approval: null, running: true } };
    case "frame":
      return { ...state, [sid]: applyFrame(cur, action.frame) };
    default:
      return state;
  }
}

export function useAgentSocket(token, onAuthFailed) {
  const [sessions, dispatch] = useReducer(reducer, {});
  const [connected, setConnected] = useState(false);
  const wsRef = useRef(null);
  const opened = useRef(new Set());
  const resync = useRef(new Set()); // sessions joined mid-run: their partial answer is fetched again at the end

  useEffect(() => {
    if (!token) return undefined;
    let stopped = false;
    let attempt = 0;
    let timer;
    const connect = () => {
      const proto = window.location.protocol === "https:" ? "wss" : "ws";
      const ws = new WebSocket(`${proto}://${window.location.host}/ws?token=${encodeURIComponent(token)}`);
      wsRef.current = ws;
      ws.onopen = () => {
        attempt = 0;
        setConnected(true);
        // After a reconnect, re-sync every session from its checkpoint.
        for (const sid of opened.current) ws.send(JSON.stringify({ op: "open", session: sid }));
      };
      ws.onmessage = (ev) => {
        const frame = JSON.parse(ev.data);
        if (!frame.session) return;
        dispatch({ type: "frame", frame });
        if (frame.type === "state" && frame.running) resync.current.add(frame.session);
        if (frame.type === "done" && resync.current.delete(frame.session))
          ws.send(JSON.stringify({ op: "open", session: frame.session }));
      };
      ws.onclose = (ev) => {
        setConnected(false);
        if (stopped) return;
        if (ev.code === 4401) { onAuthFailed(); return; }
        timer = setTimeout(connect, Math.min(10000, 500 * 2 ** attempt++));
      };
    };
    connect();
    return () => {
      stopped = true;
      clearTimeout(timer);
      if (wsRef.current) wsRef.current.close();
    };
  }, [token, onAuthFailed]);

  const send = useCallback((msg) => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(msg));
      return true;
    }
    return false;
  }, []);

  const open = useCallback((sid) => {
    opened.current.add(sid);
    dispatch({ type: "ensure", sid });
    send({ op: "open", session: sid });
  }, [send]);

  const ask = useCallback((sid, text) => {
    if (send({ op: "send", session: sid, text })) dispatch({ type: "user", sid, text });
  }, [send]);

  const approve = useCallback((sid, decisions) => {
    if (send({ op: "approve", session: sid, decisions })) dispatch({ type: "approval_sent", sid });
  }, [send]);

  const resume = useCallback((sid) => send({ op: "resume", session: sid }), [send]);
  const cancel = useCallback((sid) => send({ op: "cancel", session: sid }), [send]);

  return { sessions, connected, open, ask, approve, resume, cancel };
}
