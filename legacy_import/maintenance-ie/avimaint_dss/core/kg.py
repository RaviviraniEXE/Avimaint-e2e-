"""Knowledge graph construction + interactive rendering (pyvis).

Two graphs:
  * corpus_graph  — aggregate Component -> Fault -> Action(family) network over the
    whole corpus, edge weight = co-occurrence count. The "shape" of the data.
  * query_graph   — the entities + relations extracted for one problem (SpERT when
    on, lexicon otherwise), so the user sees exactly what the model read.

Both return a self-contained HTML string to embed with st.components.v1.html.
"""
from __future__ import annotations

import collections
import tempfile
from pathlib import Path

import networkx as nx
import pandas as pd
from pyvis.network import Network

# entity-type colours (aligned with the app palette)
TYPE_COLOR = {
    "MAINT_ITEM": "#3987e5", "FAULT": "#e66767", "ABN_PROC": "#d95926",
    "ACTION": "#199e70", "LOC": "#9085e9", "OP_CTX": "#c98500",
    "TECH_OBS": "#c98500", "OUTCOME": "#1baf7a", "REFERENCE": "#d55181",
    "Action": "#199e70",
}
_ACTION_FAMILY_COLOR = "#199e70"
_FAULT_COLOR = "#e66767"
_ITEM_COLOR = "#3987e5"


def _net(height="560px"):
    net = Network(height=height, width="100%", bgcolor="#ffffff",
                  font_color="#0f172a", directed=True, notebook=False, cdn_resources="in_line")
    net.barnes_hut(gravity=-12000, central_gravity=0.3, spring_length=120,
                   spring_strength=0.02, damping=0.5)
    net.set_options('{"nodes":{"borderWidth":0,"shadow":false,'
                    '"font":{"size":15,"face":"system-ui"}},'
                    '"edges":{"color":{"color":"#cbd5e1","highlight":"#2563eb","hover":"#64748b"},'
                    '"smooth":{"type":"dynamic"},"font":{"size":11,"color":"#64748b",'
                    '"strokeWidth":0,"align":"middle"}},'
                    '"interaction":{"hover":true,"tooltipDelay":80,"navigationButtons":false,'
                    '"multiselect":false},'
                    '"physics":{"stabilization":{"iterations":120}}}')
    return net


# JS injected into drawGraph(): click a node -> its neighbourhood stays lit, the
# rest fades; click empty space -> reset. Makes the graph explorable.
_HL_JS = """
  try {
    var _all = nodes.get({returnType:"Object"});
    var _orig = {}; for (var _i in _all) { _orig[_i] = _all[_i].color; }
    function _reset(){ var u=[]; for (var i in _all){ _all[i].color=_orig[i]; u.push(_all[i]); } nodes.update(u); }
    network.on("click", function(p){
      if (!p.nodes || p.nodes.length===0){ _reset(); return; }
      var sel = p.nodes[0];
      var conn = network.getConnectedNodes(sel); conn.push(sel);
      var u=[];
      for (var i in _all){
        var on = conn.indexOf(i) !== -1;
        _all[i].color = on ? _orig[i] : "rgba(148,163,184,0.14)";
        u.push(_all[i]);
      }
      nodes.update(u);
    });
  } catch(e){ console.log("kg-highlight:", e); }
"""


def _html(net: Network) -> str:
    try:
        html = net.generate_html(notebook=False)
    except Exception:
        tmp = Path(tempfile.mkdtemp()) / "kg.html"
        net.write_html(str(tmp), notebook=False, open_browser=False)
        html = tmp.read_text(encoding="utf-8")
    # inject neighbourhood-highlight handler where nodes/edges/network are in scope
    if "return network;" in html:
        html = html.replace("return network;", _HL_JS + "\n        return network;", 1)
    return html


def corpus_graph(df: pd.DataFrame, top_c: int = 10, top_f: int = 8,
                 min_edge: int = 3, focus_component: str | None = None) -> str:
    comp_freq = collections.Counter(x for xs in df["components"] for x in xs
                                    if x and x != "(unspecified)")
    fault_freq = collections.Counter(x for xs in df["faults"] for x in xs
                                     if x and x != "(unspecified)")

    if focus_component:
        # ego graph: the chosen component -> its faults -> their actions
        top_comps = {focus_component}
        # faults that co-occur with the focus component (any of the top faults)
        top_faults = set(f for f, _ in fault_freq.most_common(max(top_f, 12)))
        min_edge = 1
    else:
        top_comps = {c for c, _ in comp_freq.most_common(top_c)}
        top_faults = {f for f, _ in fault_freq.most_common(top_f)}

    cf = collections.Counter()   # (component, fault)
    fa = collections.Counter()   # (fault, action_family)
    for r in df.itertuples(index=False):
        comps = set(r.components) & top_comps
        faults = set(r.faults) & top_faults
        fam = r.action_family
        for c in comps:
            for f in faults:
                cf[(c, f)] += 1
        if fam and fam != "Other":
            # in focus mode, only count actions for rows that touch the focus component
            if focus_component and focus_component not in set(r.components):
                continue
            for f in faults:
                fa[(f, fam)] += 1

    net = _net("620px")
    used_faults = set()
    for (c, f), w in cf.items():
        if w < min_edge:
            continue
        focus = (c == focus_component)
        net.add_node("C:" + c, label=c, color=_ITEM_COLOR, shape="dot",
                     borderWidth=3 if focus else 0,
                     size=(22 if focus else 12) + comp_freq[c] ** 0.5,
                     title=f"component · {comp_freq[c]} work orders")
        net.add_node("F:" + f, label=f, color=_FAULT_COLOR, shape="dot",
                     size=12 + fault_freq[f] ** 0.5, title=f"fault · {fault_freq[f]} work orders")
        used_faults.add(f)
        net.add_edge("C:" + c, "F:" + f, value=w, title=f"{w} work orders", label=str(w))
    for (f, a), w in fa.items():
        if w < min_edge or f not in used_faults:
            continue
        net.add_node("A:" + a, label=a, color=_ACTION_FAMILY_COLOR, shape="square",
                     size=16, title="action family")
        net.add_edge("F:" + f, "A:" + a, value=w, title=f"{w} work orders", label=str(w))
    return _html(net)


def query_graph(entities: list[dict], relations: list[dict]) -> str:
    net = _net("420px")
    # unique node per (type,text)
    def nid(e):
        return f'{e.get("type")}:{e.get("text","").lower()}'
    seen = set()
    for e in entities:
        i = nid(e)
        if i in seen:
            continue
        seen.add(i)
        c = TYPE_COLOR.get(e.get("type"), "#94a3b8")
        net.add_node(i, label=e.get("text", ""), color=c, shape="dot", size=18,
                     title=f'{e.get("type")} · score {e.get("score", "")}')
    for r in relations:
        h = f'{r.get("head_type")}:{r.get("head_text","").lower()}'
        t = f'{r.get("tail_type")}:{r.get("tail_text","").lower()}'
        if h not in seen:
            net.add_node(h, label=r.get("head_text", ""), color=TYPE_COLOR.get(r.get("head_type"), "#94a3b8"), size=18)
            seen.add(h)
        if t not in seen:
            net.add_node(t, label=r.get("tail_text", ""), color=TYPE_COLOR.get(r.get("tail_type"), "#94a3b8"), size=18)
            seen.add(t)
        net.add_edge(h, t, label=r.get("type", ""), title=r.get("type", ""))
    if not entities:
        net.add_node("none", label="no entities extracted", color="#94a3b8")
    return _html(net)

