<!-- projectos:begin v1 -->
## ProjectOS (read first)
This project keeps shared agent memory in `.projectos/` (purpose, map, state, attempts, decisions, checkpoints) for every agent: Claude, Codex, Antigravity, humans.
1. Start: read `.projectos/STATE.md` and `PURPOSE.md` (or `pos brief`). Hooks may have injected this already.
2. Code questions: `.projectos/MAP.md` first, raw files last.
3. Before a new approach: `pos attempt list --grep <kw>`. Never repeat a FAILED attempt without stating what differs.
4. Log outcomes (`pos attempt add`, `pos decide`). After each finished sub-task: update STATE.md, then `pos checkpoint -m "..."`.
Rules: `.projectos/PROTOCOL.md`. No `pos` on this machine? `python .projectos/kit/install.py` if `.projectos/kit/` exists (full setup), else edit the markdown files directly, same formats.
<!-- projectos:end -->

## graphify-context

Anchor ID: graphify-hub-main-v1
Primary shared graph:
- G:\Other computers\My PC\Desktop\graphify total context\graphify-out\GRAPH_REPORT.md
- G:\Other computers\My PC\Desktop\graphify total context\graphify-out\graph.json
Secondary local graph:
- G:\My Drive\PhD & Research\Temporal GNN\graphify-out\GRAPH_REPORT.md
- G:\My Drive\PhD & Research\Temporal GNN\graphify-out\graph.json
Project notes:
- G:\Other computers\My PC\Desktop\graphify total context\projects\temporal-gnn\notes

Rules:
- Startup default: Graphify-first, token-efficient (headings/summary first).
- On project open: check if mapped context exists in shared hub and attach to it.
- Read shared graph first, local graph second, notes third, raw files last.
- If shared and local conflict, prefer local facts for repo-specific behavior.
- Record major actions to shared acts ledger.
