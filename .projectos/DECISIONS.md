# Decisions - Temporal GNN
Append-only ADR-lite. Supersede instead of editing history.

    ### D-001 | YYYY-MM-DD | agent | ACTIVE | short title
    - decision: what was chosen
    - why: reasons / evidence
    - alternatives: what was rejected and why
    - supersedes: D-000 (optional)

STATUS: PROPOSED | ACTIVE | SUPERSEDED by D-xxx | REVERTED


### D-001 | 2026-10-02 | claude | PROPOSED | Project purpose = interpretable real-time generative population model (TACK)
- decision: fill PURPOSE.md around the user's 2026-10-01 goal ("modellare l'evoluzione nel tempo di una popolazione di cellule in modo generativo e quasi continuo in modo interpretable") with TACK as the method line; tgl-grn and the cfm_grn_v2 engine programme become non-goals/baselines.
- why: 17-agent analysis (docs/ANALISI_E_APPROCCIO_2026-10-01.md); user said "prova tutto quello che hai suggerito" (2026-10-02); calibration gate and P1' passed the same day; erythroid probe of the simple age prior was NEGATIVE vs L-mix (docs/superpowers/decisions.md).
- alternatives: keep RA-FM/cfm_grn_v2 as core (rejected: interpretable 0/5, fresh-data negative + tie); program-space SDE as core (deferred to year 2).
- status note: needs the user's explicit yes before ACTIVE.

### D-002 | 2026-10-02 | claude | ACTIVE | D-001 confirmed
- decision: D-001 (project purpose = TACK line) becomes ACTIVE.
- why: the user answered the three open decisions (simulation recovery, pilot GSE236517 download, PURPOSE.md yes) with "vai" (2026-10-02).
- supersedes: D-001 status PROPOSED
