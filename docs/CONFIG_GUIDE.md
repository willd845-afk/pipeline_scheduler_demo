# Pipeline Scheduler — Configuration File Guide

This document describes the format and usage of the two types of configuration
files accepted by the pipeline scheduler: the YAML pipeline config file and the
JSON state file.

---

## Table of Contents

1. [YAML Config File](#1-yaml-config-file)
   - 1.1 [File location](#11-file-location)
   - 1.2 [Top-level structure](#12-top-level-structure)
   - 1.3 [meta](#13-meta)
   - 1.4 [instructions](#14-instructions)
   - 1.5 [bypass_types](#15-bypass_types)
   - 1.6 [pipeline](#16-pipeline)
   - 1.7 [defaults](#17-defaults)
   - 1.8 [rules](#18-rules)
   - 1.9 [Complete example](#19-complete-example)
2. [JSON State File](#2-json-state-file)
   - 2.1 [Overview](#21-overview)
   - 2.2 [Structure](#22-structure)
   - 2.3 [grid_data](#23-grid_data)
   - 2.4 [instructions](#24-instructions)
   - 2.5 [bypass_annotations](#25-bypass_annotations)
   - 2.6 [Complete example](#26-complete-example)
3. [Using the Files](#3-using-the-files)
   - 3.1 [Loading a YAML config](#31-loading-a-yaml-config)
   - 3.2 [Saving and loading state](#32-saving-and-loading-state)
   - 3.3 [Using a start state](#33-using-a-start-state)
4. [Validation Rules and Error Messages](#4-validation-rules-and-error-messages)
5. [Quick Reference](#5-quick-reference)

---

## 1. YAML Config File

### 1.1 File Location

The application looks for a config file at the following hard-coded path
relative to the project root:

```
config/config.yaml
```

This file is loaded automatically when the server starts. A different config
can be loaded at runtime using the **Import Config** button in the interface,
which accepts any `.yaml` or `.yml` file from your computer.

---

### 1.2 Top-level Structure

A config file has six top-level sections. The `meta`, `instructions`, and
`pipeline` sections are required. The others are optional but recommended.

```yaml
meta:           # Required — name and description
instructions:   # Required — instruction set definition
bypass_types:   # Optional — hazard annotation categories
pipeline:       # Required — stage names and column capacities
defaults:       # Optional — starting grid size and state file
rules:          # Optional — validation rule definitions
```

---

### 1.3 `meta`

Human-readable metadata displayed in the interface after the config is loaded.

```yaml
meta:
  name: "My Pipeline Config"
  description: "A short description of this configuration"
```

| Field | Type | Required | Description |
|---|---|---|---|
| `name` | string | Yes | Short name shown next to the Import Config button |
| `description` | string | No | Longer description for documentation purposes |

---

### 1.4 `instructions`

Defines the instruction set that appears in the row label dropdowns.
Each key is the instruction mnemonic as it will appear in the UI.

```yaml
instructions:
  <mnemonic>:
    type:     <INSTRUCTION_TYPE>
    operands: [<operand1>, <operand2>, ...]
    format:   "<assembly syntax string>"
```

#### Fields

| Field | Type | Required | Description |
|---|---|---|---|
| `type` | string | Yes | One of the valid instruction type strings (see below) |
| `operands` | list of strings | Yes | Ordered list of operand names; must contain at least one |
| `format` | string | Yes | Assembly display template; operand names are substituted with their values |

#### Valid instruction types

| Type string | Meaning |
|---|---|
| `R_TYPE` | Register–register operation (e.g. add, mul) |
| `I_TYPE` | Immediate or load operation (e.g. addi, lw) |
| `I_STORE` | Store operation (e.g. sw) |
| `B_TYPE` | Branch operation (e.g. bne) |
| `J_TYPE` | Jump-and-link (e.g. jal) |
| `JR_TYPE` | Jump-register (e.g. jr) |

#### Operand naming conventions

Operand names are arbitrary strings but the following conventions are
recommended for clarity:

| Name | Conventional meaning |
|---|---|
| `rd` | Destination register |
| `rs1` | First source register |
| `rs2` | Second source register |
| `imm` | Immediate value |

#### Format string syntax

The `format` string is an assembly template. Operand names are replaced with
their concrete values when an instruction is rendered. Write the operand names
exactly as they appear in the `operands` list.

Special patterns for load/store:

```yaml
# Load: destination first, then offset(base)
format: "lw rd, imm(rs1)"

# Store: source first, then offset(base)
format: "sw rs2, imm(rs1)"
```

#### Example

```yaml
instructions:
  add:
    type:     R_TYPE
    operands: [rd, rs1, rs2]
    format:   "add rd, rs1, rs2"

  addi:
    type:     I_TYPE
    operands: [rd, rs1, imm]
    format:   "addi rd, rs1, imm"

  lw:
    type:     I_TYPE
    operands: [rd, rs1, imm]
    format:   "lw rd, imm(rs1)"

  sw:
    type:     I_STORE
    operands: [rs2, rs1, imm]
    format:   "sw rs2, imm(rs1)"

  jr:
    type:     JR_TYPE
    operands: [rs1]
    format:   "jr rs1"
```

---

### 1.5 `bypass_types`

Defines the hazard/bypass annotation categories that appear in the bypass
tool dropdown. Each entry creates one coloured arrow type that users can
draw between grid cells.

If this section is absent the bypass tool dropdown will be empty, but the
tool itself will still be accessible.

```yaml
bypass_types:
  - name:        "<identifier>"
    color:       "<hex colour>"
    description: "<full label>"
```

| Field | Type | Required | Description |
|---|---|---|---|
| `name` | string | Yes | Short identifier shown in the dropdown and stored with annotations |
| `color` | string | Yes | Hex colour string including the `#` prefix, e.g. `#cc0000` |
| `description` | string | Yes | Full label shown alongside the name in the dropdown |

#### Example

```yaml
bypass_types:
  - name:        RAW
    color:       "#cc0000"
    description: "Read After Write"

  - name:        WAR
    color:       "#cc6600"
    description: "Write After Read"

  - name:        WAW
    color:       "#9900cc"
    description: "Write After Write"

  - name:        Control
    color:       "#006600"
    description: "Control Dependency"

  - name:        Structural
    color:       "#0066cc"
    description: "Structural Hazard"
```

---

### 1.6 `pipeline`

Defines the pipeline stages that appear as draggable blocks in the palette
and sets the maximum number of times each stage may appear in a single column.

```yaml
pipeline:
  stages:
    <stage_name>: <capacity>
```

#### Capacity values

| Value | Meaning |
|---|---|
| `0` | Unbounded — stage may appear any number of times in a column |
| `1` | At most once per column |
| `2` | At most twice per column |
| `N` | At most N times per column |

The order of stages in this section determines the order they appear in the
palette. Stages are displayed left-to-right in palette order.

#### Example

```yaml
pipeline:
  stages:
    F:  1    # Fetch      — at most once per column
    D:  1    # Decode     — at most once per column
    i:  0    # Stall      — unbounded (may appear any number of times)
    I:  1    # Issue      — at most once per column
    Y0: 1    # Execute 0  — at most once per column
    Y1: 1    # Execute 1  — at most once per column
    Y2: 1    # Execute 2  — at most once per column
    Y3: 1    # Execute 3  — at most once per column
    W:  1    # Write-back — at most once per column
    r:  0    # Repeat     — unbounded
    C:  1    # Commit     — at most once per column
    X:  1    # Exception  — at most once per column
```

---

### 1.7 `defaults`

Sets the initial grid dimensions and optionally references a JSON state file
to pre-populate the grid when the config is loaded.

```yaml
defaults:
  rows:        10
  cols:        10
  start_state: "./config/start_example.json"
```

| Field | Type | Required | Description |
|---|---|---|---|
| `rows` | integer | No | Initial number of grid rows. Defaults to 10 if absent |
| `cols` | integer | No | Initial number of grid columns. Defaults to 10 if absent |
| `start_state` | string (file path) | No | Path to a JSON state file to load as the initial grid content. Path is relative to the project root. Omit or set to `null` to start with an empty grid |

If `start_state` is provided and the file exists, its `rows` and `cols`
values override the `defaults.rows` and `defaults.cols` values above.

---

### 1.8 `rules`

Defines the validation rules that are checked after every grid interaction.
Each rule appears as a checkbox item in the Pipeline Rules panel.

Only the `column_constraint` rule type is currently supported. Future
versions may introduce additional types; unknown types are skipped with
a warning rather than causing an error.

```yaml
rules:
  - name:        "<display name>"
    description: "<tooltip/panel description>"
    enabled:     true
    type:        "column_constraint"
    logic:
      scope:  "column"
      target: "grid_data"
      constraint:
        group_by: "stage"
        limit:
          from: "pipeline.stages"
          key:  "stage"
        ignore_if:
          capacity_equals: 0
        violation_message: >
          Stage '{stage}' exceeds its allowed capacity ({capacity})
          in column {col}.
```

#### Top-level rule fields

| Field | Type | Required | Description |
|---|---|---|---|
| `name` | string | Yes | Display name shown in the rules panel |
| `description` | string | Yes | Explanation shown next to the checkbox |
| `enabled` | boolean | Yes | Whether the rule is active when the config is loaded |
| `type` | string | Yes | Rule implementation type; currently only `column_constraint` |
| `logic` | mapping | Yes (for column_constraint) | Declarative logic definition |

#### `logic` fields for `column_constraint`

| Field | Description |
|---|---|
| `scope` | Always `"column"` — checks are performed per column |
| `target` | Always `"grid_data"` — evaluates placed blocks |
| `constraint.group_by` | Always `"stage"` — groups blocks by their type name |
| `constraint.limit.from` | Always `"pipeline.stages"` — reads capacities from the pipeline section |
| `constraint.limit.key` | Always `"stage"` |
| `constraint.ignore_if.capacity_equals` | Integer; stages with this exact capacity are exempt. Set to `0` to make unbounded stages skip checking |
| `constraint.violation_message` | Template string. Use `{stage}`, `{col}`, and `{capacity}` as placeholders |

#### Example

```yaml
rules:
  - name: "Unique Stage Per Column"
    description: >
      Ensures that each pipeline stage appears at most once per column,
      unless the stage capacity is set to 0 (unbounded).
    enabled: true
    type: "column_constraint"
    logic:
      scope:  "column"
      target: "grid_data"
      constraint:
        group_by: "stage"
        limit:
          from: "pipeline.stages"
          key:  "stage"
        ignore_if:
          capacity_equals: 0
        violation_message: >
          Stage '{stage}' exceeds its allowed capacity ({capacity})
          in column {col}.
```

---

### 1.9 Complete Example

```yaml
meta:
  name: "Default Pipeline Configuration"
  description: "Baseline configuration for teaching pipelining concepts"

instructions:
  add:
    type:     R_TYPE
    operands: [rd, rs1, rs2]
    format:   "add rd, rs1, rs2"

  addi:
    type:     I_TYPE
    operands: [rd, rs1, imm]
    format:   "addi rd, rs1, imm"

  mul:
    type:     R_TYPE
    operands: [rd, rs1, rs2]
    format:   "mul rd, rs1, rs2"

  bne:
    type:     B_TYPE
    operands: [rs1, rs2, imm]
    format:   "bne rs1, rs2, imm"

  jr:
    type:     JR_TYPE
    operands: [rs1]
    format:   "jr rs1"

  lw:
    type:     I_TYPE
    operands: [rd, rs1, imm]
    format:   "lw rd, imm(rs1)"

  sw:
    type:     I_STORE
    operands: [rs2, rs1, imm]
    format:   "sw rs2, imm(rs1)"

  jal:
    type:     J_TYPE
    operands: [rd, imm]
    format:   "jal rd, imm"

bypass_types:
  - name:        RAW
    color:       "#cc0000"
    description: "Read After Write"

  - name:        WAR
    color:       "#cc6600"
    description: "Write After Read"

  - name:        WAW
    color:       "#9900cc"
    description: "Write After Write"

  - name:        Control
    color:       "#006600"
    description: "Control Dependency"

  - name:        Structural
    color:       "#0066cc"
    description: "Structural Hazard"

pipeline:
  stages:
    F:  1
    D:  1
    i:  0
    I:  1
    Y0: 1
    Y1: 1
    Y2: 1
    Y3: 1
    W:  1
    r:  0
    C:  1
    X:  1

defaults:
  rows:        10
  cols:        10
  start_state: "./config/start_example.json"

rules:
  - name: "Unique Stage Per Column"
    description: >
      Ensures that each pipeline stage appears at most once per column,
      unless the stage capacity is set to 0 (unbounded).
      This models structural limits of pipeline hardware.
    enabled: true
    type: "column_constraint"
    logic:
      scope:  "column"
      target: "grid_data"
      constraint:
        group_by: "stage"
        limit:
          from: "pipeline.stages"
          key:  "stage"
        ignore_if:
          capacity_equals: 0
        violation_message: >
          Stage '{stage}' exceeds its allowed capacity ({capacity})
          in column {col}.
```

---

## 2. JSON State File

### 2.1 Overview

A JSON state file captures a complete snapshot of the grid at a point in time.
It is used in two ways:

- **Manual save/load**: The Save State and Load State buttons in the interface
  export and import state files directly.
- **Default start state**: A state file can be referenced in the YAML config
  under `defaults.start_state` to pre-populate the grid when the config loads.

State files do **not** contain configuration data such as instruction
definitions, bypass types, or rule definitions. Those always come from the
YAML config file.

---

### 2.2 Structure

```json
{
  "grid_data":           { },
  "instructions":        { },
  "rows":                10,
  "cols":                10,
  "bypass_annotations":  [ ]
}
```

| Field | Type | Description |
|---|---|---|
| `grid_data` | object | Maps `"row,col"` coordinate strings to block type strings |
| `instructions` | object | Maps row number strings to instruction objects |
| `rows` | integer | Number of grid rows to restore |
| `cols` | integer | Number of grid columns to restore |
| `bypass_annotations` | array | List of bypass annotation objects |

All fields are optional. Missing fields are replaced with empty defaults so
that a partial state file is safe to load.

---

### 2.3 `grid_data`

Each key is a string in the format `"row,col"` where both values are
zero-based integers. Each value is a stage name string matching one of the
stages defined in the YAML config's `pipeline.stages` section.

```json
"grid_data": {
  "0,0": "F",
  "0,1": "D",
  "0,2": "I",
  "0,3": "W",
  "1,1": "F",
  "1,2": "D",
  "1,3": "i",
  "1,4": "I"
}
```

Blocks outside the current `rows`/`cols` bounds are preserved in the saved
file and will reappear if the grid is later enlarged.

---

### 2.4 `instructions`

Each key is a row number as a string. Each value is an instruction object
with a `name` field matching one of the instruction mnemonics defined in the
YAML config, and an `operands` object mapping operand names to their values.

```json
"instructions": {
  "0": {
    "name": "add",
    "operands": {
      "rd":  "r1",
      "rs1": "r2",
      "rs2": "r3"
    }
  },
  "1": {
    "name": "lw",
    "operands": {
      "rd":  "r4",
      "rs1": "r5",
      "imm": "8"
    }
  }
}
```

To represent a row with no instruction assigned, omit it from the object
entirely. Do not store an entry with an empty `name`.

Operand values are always stored as strings even when the value is a number
(e.g. immediate values).

---

### 2.5 `bypass_annotations`

An array of annotation objects. Each object describes one drawn bypass arrow
connecting two grid cells.

```json
"bypass_annotations": [
  {
    "annotation_type": "RAW",
    "color":           "#cc0000",
    "source": { "row": 0, "col": 3 },
    "target": { "row": 1, "col": 2 }
  }
]
```

| Field | Type | Description |
|---|---|---|
| `annotation_type` | string | Must match a `name` in the loaded config's `bypass_types` section |
| `color` | string | Hex colour string; should match the colour of the named bypass type |
| `source` | object | `{"row": int, "col": int}` — origin cell (circled end) |
| `target` | object | `{"row": int, "col": int}` — destination cell (arrowhead end) |

If `annotation_type` does not match any type in the currently loaded config,
the annotation is still drawn using the stored `color` value.

---

### 2.6 Complete Example

```json
{
  "grid_data": {
    "0,0": "F",
    "0,1": "D",
    "0,2": "I",
    "0,3": "Y0",
    "0,4": "W",
    "1,1": "F",
    "1,2": "D",
    "1,3": "i",
    "1,4": "I",
    "1,5": "Y0",
    "1,6": "W",
    "2,2": "F",
    "2,3": "D",
    "2,4": "i",
    "2,5": "i",
    "2,6": "I",
    "2,7": "Y0",
    "2,8": "W"
  },
  "instructions": {
    "0": {
      "name": "add",
      "operands": { "rd": "r1", "rs1": "r2", "rs2": "r3" }
    },
    "1": {
      "name": "addi",
      "operands": { "rd": "r4", "rs1": "r1", "imm": "1" }
    },
    "2": {
      "name": "lw",
      "operands": { "rd": "r5", "rs1": "r1", "imm": "0" }
    }
  },
  "rows": 6,
  "cols": 12,
  "bypass_annotations": [
    {
      "annotation_type": "RAW",
      "color":           "#cc0000",
      "source": { "row": 0, "col": 4 },
      "target": { "row": 1, "col": 4 }
    },
    {
      "annotation_type": "RAW",
      "color":           "#cc0000",
      "source": { "row": 0, "col": 4 },
      "target": { "row": 2, "col": 6 }
    }
  ]
}
```

---

## 3. Using the Files

### 3.1 Loading a YAML Config

**At startup (automatic):**
Place the file at `config/config.yaml` relative to the project root. The
server loads this file automatically on startup. If the file is missing or
contains errors the server will raise an exception and exit.

**At runtime (Import Config button):**
1. Click **Import Config** in the controls panel
2. Select a `.yaml` or `.yml` file from your computer
3. The file is validated on the server; any errors are shown in an alert
4. On success the palette, bypass tool dropdown, rules panel, and grid are
   all updated to reflect the new config without a page reload
5. The config name is displayed next to the Import Config button

The grid content (blocks, instructions, annotations) is preserved when a
new config is imported. Only the palette blocks, instruction dropdown options,
bypass type dropdown, and rule definitions change.

---

### 3.2 Saving and Loading State

**Save State:**
Click **Save State** to download the current grid as a `pipeline_state.json`
file. This file captures:
- All placed blocks and their positions
- All instruction assignments
- All bypass annotations
- The current grid dimensions

It does **not** capture which config file is loaded.

**Load State:**
Click **Load State** to select a previously saved `.json` file. The grid,
instructions, and annotations are restored to the saved values. The currently
loaded config is unchanged.

---

### 3.3 Using a Start State

To have a grid pre-populated whenever a config is loaded:

1. Create and save a JSON state file (use Save State from the interface)
2. Place the file somewhere accessible, e.g. `./config/start_example.json`
3. Reference it in the YAML config:

```yaml
defaults:
  rows:        8
  cols:        14
  start_state: "./config/start_example.json"
```

4. The `rows` and `cols` values inside the JSON file take precedence over
   the `defaults.rows` and `defaults.cols` values in the YAML.

The path is relative to the project root. Absolute paths are also accepted.
If the file does not exist or contains invalid JSON, the config still loads
successfully but the grid starts empty, and a warning is returned in the
validation step.

---

## 4. Validation Rules and Error Messages

The server validates the YAML file before applying it. Below are the checks
performed and the error messages they produce.

### Instructions

| Condition | Error message |
|---|---|
| `instructions` block is missing or empty | `'instructions' block is empty or missing` |
| An instruction entry is not a YAML mapping | `Instruction 'X': must be a mapping, got <type>` |
| `type` field is missing | `Instruction 'X': missing required field 'type'` |
| `type` value is not in the valid set | `Instruction 'X': invalid type 'Y'; must be one of [...]` |
| `operands` field is missing | `Instruction 'X': missing required field 'operands'` |
| `operands` is not a non-empty list | `Instruction 'X': 'operands' must be a non-empty list` |
| `format` field is missing | `Instruction 'X': missing required field 'format'` |

### Bypass types

| Condition | Error message |
|---|---|
| An entry is not a YAML mapping | `bypass_types[N]: must be a mapping, got <type>` |
| `name`, `color`, or `description` is missing | `bypass_types[N]: missing required field '<field>'` |

### Pipeline stages

| Condition | Error message |
|---|---|
| `pipeline.stages` block is missing or empty | `'pipeline.stages' block is empty or missing` |
| A capacity value is not a non-negative integer | `Stage 'X': capacity must be a non-negative integer, got 'Y'` |
| A capacity value is negative | `Stage 'X': capacity must be >= 0, got N` |

### Rules

| Condition | Error message |
|---|---|
| An entry is not a YAML mapping | `rules[N]: must be a mapping, got <type>` |
| `name`, `description`, `enabled`, or `type` is missing | `rules[N]: missing required field '<field>'` |
| A rule `type` is not in the known set | Warning issued; rule is skipped (not a validation error) |

### Start state

| Condition | Error message |
|---|---|
| The referenced file does not exist | `defaults.start_state: path '...' does not exist` |
| The file contains invalid JSON | `defaults.start_state: '...' is not valid JSON — <detail>` |

---

## 5. Quick Reference

### YAML config skeleton

```yaml
meta:
  name: ""
  description: ""

instructions:
  <mnemonic>:
    type:     <R_TYPE|I_TYPE|I_STORE|B_TYPE|J_TYPE|JR_TYPE>
    operands: [<operand>, ...]
    format:   "<assembly template>"

bypass_types:
  - name:        <identifier>
    color:       "<#rrggbb>"
    description: "<label>"

pipeline:
  stages:
    <stage_name>: <capacity_integer>

defaults:
  rows:        10
  cols:        10
  start_state: null

rules:
  - name:        "<display name>"
    description: "<panel description>"
    enabled:     true
    type:        "column_constraint"
    logic:
      scope:  "column"
      target: "grid_data"
      constraint:
        group_by: "stage"
        limit:
          from: "pipeline.stages"
          key:  "stage"
        ignore_if:
          capacity_equals: 0
        violation_message: >
          Stage '{stage}' exceeds its allowed capacity ({capacity})
          in column {col}.
```

### JSON state file skeleton

```json
{
  "grid_data": {
    "<row>,<col>": "<stage_name>"
  },
  "instructions": {
    "<row>": {
      "name": "<mnemonic>",
      "operands": {
        "<operand_name>": "<value>"
      }
    }
  },
  "rows": 10,
  "cols": 10,
  "bypass_annotations": [
    {
      "annotation_type": "<bypass_type_name>",
      "color": "<#rrggbb>",
      "source": { "row": 0, "col": 0 },
      "target": { "row": 0, "col": 0 }
    }
  ]
}
```