### Conversation Context & Progress Checkpoint
**Date:** May 29, 2026  
**Project:** Pipeline Scheduler Demo (Declarative YAML Rules, Dependency Graph, RAW Bypass Validation)  
**Goal:** Run the self-contained diagnostic script (`Debugging/diagnostic.py`) successfully to identify why valid bypasses do not cancel RAW hazards.

---

### 1. Where We Left Off (The Big Picture)
We resolved the initial `404 Not Found` errors by unconditionally registering the debug blueprint (`debug_bp`) in `app.py` and binding it to the active `scheduler` instance via `init_debug_blueprint(scheduler)`. 

We then encountered server-side `500 Internal Server Errors` because the attribute names in `pipeline/debug_routes.py` did not match the underlying properties in `pipeline/scheduler.py` and `pipeline/instructions.py`. We have fixed two of these, and the remaining two require adding methods to `scheduler.py`.

---

### 2. Recent Fixes (Applied)
*   **Resolved Blueprint Registration (404s):** Removed the `if app.debug` conditional wrapper. The debug routes are now unconditionally registered at `/api/debug/*`.
*   **Resolved `instruction-formats` 500:** Modified the route to call `_scheduler.config.to_instruction_formats()` instead of trying to access `instruction_formats` directly as an attribute.
*   **Resolved `stage-columns` 500:** Added an introspection loop in the route to search `Instruction` objects for common dictionary names (`stages`, `stage_map`, `col_to_stage`, `placement`, etc.) to find where stage-to-column mappings are stored.

---

### 3. Immediate Blockers & Remaining 500 Errors
When the diagnostic script runs, it still fails on the final stages because the following methods do not yet exist in `PipelineScheduler` (`pipeline/scheduler.py`):

1.  **`check_bypass_bounds`** is missing:
    *   *Route:* `/api/debug/bypass-validation`
    *   *Error:* `AttributeError: 'PipelineScheduler' object has no attribute 'check_bypass_bounds'`
2.  **`evaluate_rules_trace`** is missing:
    *   *Route:* `/api/debug/rule-trace`
    *   *Error:* `AttributeError: 'PipelineScheduler' object has no attribute 'evaluate_rules_trace'`

---

### 4. Next Steps for Next Session
To complete the implementation and clear the final 500 errors, the developer must:
*   Inspect **`pipeline/scheduler.py`** to see:
    *   How `bypass_annotations` are represented and stored.
    *   The `__init__` constructor and its existing attributes.
    *   How rule evaluation is currently structured.
*   Inspect **`pipeline/instructions.py`** to identify:
    *   The exact name of the dictionary mapping columns to stage strings (to replace the temporary introspection loop in `stage-columns`).
*   Implement `check_bypass_bounds` and `evaluate_rules_trace` inside `scheduler.py` based on the YAML rule spec.

***

*I have saved this summary to my long-term memory. When you are ready to resume, you can simply paste this card, along with the contents of your `pipeline/scheduler.py` and `pipeline/instructions.py` files, and we will write the missing methods to make the diagnostics pass!*