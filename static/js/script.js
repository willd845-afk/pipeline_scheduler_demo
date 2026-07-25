// ── Flask-injected data ────────────────────────────────────────────────────
// Declared with 'let' so importConfig() can reassign them after a hot reload.
let INSTRUCTION_FORMATS = window.FLASK_DATA.instructionFormats;
let INSTRUCTION_NAMES   = Object.keys(INSTRUCTION_FORMATS);
let BYPASS_TYPES        = window.FLASK_DATA.bypassTypes;   // was PIPELINE_TYPES
let BLOCK_TYPES         = window.FLASK_DATA.blockTypes;    // was a hard-coded array

// ── Grid state ────────────────────────────────────────────────────────────
let currentRows = 10;
let currentCols = 10;
let gridData    = {};
let instructions = {};

// ── Rule / violation state ────────────────────────────────────────────────
let currentViolations = [];
let rulesInfo         = [];

// ── Bypass annotation mode state ──────────────────────────────────────────
let bypassAnnotationMode = false;
let selectedBypassType   = null;
let bypassSource         = null;
let bypassAnnotations    = [];
let bypassMenuOpen       = false;
let bypassDragging       = false;

// Maps bypass annotation array index → hex color string for violation highlights.
// Populated by updateViolationUI(); consumed by renderBypassAnnotations().
// Reset to {} when violations are cleared.
let currentAnnotationHighlights = {};

/**
 * Convert a hex colour string to an {r, g, b} object.
 * Used to build rgba() values for CSS variables and the ghost circle.
 *
 * @param {string} hex - e.g. '#cc0000'
 * @returns {{r: number, g: number, b: number}}
 */
function hexToRgb(hex) {
    return {
        r: parseInt(hex.slice(1, 3), 16),
        g: parseInt(hex.slice(3, 5), 16),
        b: parseInt(hex.slice(5, 7), 16)
    };
}

/**
 * Initialize the block palette with draggable block elements.
 * Creates DOM elements for each block type that can be dragged onto the grid.
 */
function initPalette() {
    const palette = document.getElementById('palette');
    palette.innerHTML = '';
    BLOCK_TYPES.forEach(blockType => {
        const block = document.createElement('div');
        block.className = 'palette-block';
        block.textContent = blockType;
        block.draggable = true;
        block.dataset.blockType = blockType;

        block.addEventListener('dragstart', (e) => {
            if (bypassAnnotationMode) exitPipelineMode();
            e.dataTransfer.setData('blockType', blockType);
            e.dataTransfer.setData('source', 'palette');
            block.classList.add('dragging');
        });

        block.addEventListener('dragend', (e) => {
            block.classList.remove('dragging');
        });

        // Also exit if the user merely clicks a palette block (without dragging)
        block.addEventListener('click', () => {          // NEW
            if (bypassAnnotationMode) exitPipelineMode();
        });

        palette.appendChild(block);
    });
}
/**
 * Load rules information from the server and populate the rules panel.
 * Fetches the current rule configuration and renders checkboxes for each rule.
 * @async
 */
async function loadRules() {
    const response = await fetch('/api/rules');
    const data = await response.json();
    rulesInfo = data.rules || [];
    renderRulesPanel();
}

/**
 * Render the rules panel with checkboxes for enabling/disabling rules.
 * Updates the UI to reflect current rule states.
 */
function renderRulesPanel() {
    const rulesList = document.getElementById('rules-list');
    rulesList.innerHTML = '';

    rulesInfo.forEach(rule => {
        const ruleDiv = document.createElement('div');
        ruleDiv.className = 'rule-item' + (rule.enabled ? '' : ' disabled');

        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.checked = rule.enabled;
        checkbox.onchange = () => toggleRule(rule.name, checkbox.checked);

        const content = document.createElement('span');
        content.innerHTML = `<strong>${rule.name}:</strong> ${rule.description}`;

        ruleDiv.appendChild(checkbox);
        ruleDiv.appendChild(content);
        rulesList.appendChild(ruleDiv);
    });
}

/**
 * Toggle a specific rule on or off.
 *
 * @async
 * @param {string} ruleName - The name of the rule to toggle
 * @param {boolean} enabled - Whether to enable or disable the rule
 */
async function toggleRule(ruleName, enabled) {
    await fetch('/api/rules', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: ruleName, enabled: enabled })
    });

    // Update local rules info
    const rule = rulesInfo.find(r => r.name === ruleName);
    if (rule) {
        rule.enabled = enabled;
    }

    renderRulesPanel();
    await checkRules();
}

/**
 * Enable or disable all rules at once.
 *
 * @async
 * @param {boolean} enabled - Whether to enable or disable all rules
 */
async function toggleAllRules(enabled) {
    await fetch('/api/rules', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ all: enabled })
    });

    // Update local rules info
    rulesInfo.forEach(rule => {
        rule.enabled = enabled;
    });

    renderRulesPanel();
    await checkRules();
}

/**
 * Create an instruction editor UI for a specific row.
 * Generates a dropdown and operand inputs for instruction configuration.
 *
 * @param {number} row - The row index for this editor
 * @returns {HTMLElement} The container element with instruction editor controls
 */
function createInstructionEditor(row) {
    const container = document.createElement('div');
    container.className = 'operand-container';

    const select = document.createElement('select');
    const defaultOption = document.createElement('option');
    defaultOption.value = '';
    defaultOption.textContent = '-- Select --';
    select.appendChild(defaultOption);

    INSTRUCTION_NAMES.forEach(name => {
        const option = document.createElement('option');
        option.value = name;
        option.textContent = name;
        select.appendChild(option);
    });

    if (instructions[row]) {
        select.value = instructions[row].name;
    }

    select.addEventListener('change', (e) => {
        updateInstruction(row, e.target.value);
    });

    container.appendChild(select);

    if (instructions[row] && instructions[row].name) {
        addOperandInputs(container, row, instructions[row]);
    }

    return container;
}

/**
 * Add operand input fields to an instruction editor container.
 * Handles special formatting for load/store instructions.
 *
 * @param {HTMLElement} container - The container to add inputs to
 * @param {number} row - The row index
 * @param {Object} instruction - The instruction object with name and operands
 */
function addOperandInputs(container, row, instruction) {
    const format = INSTRUCTION_FORMATS[instruction.name];
    if (!format) return;

    const operands = format.operands;

    if (instruction.name === 'lw' || instruction.name === 'sw') {
        if (instruction.name === 'lw') {
            addOperandInput(container, row, 'rd', instruction.operands.rd || '');
            container.appendChild(document.createTextNode(', '));
            addOperandInput(container, row, 'imm', instruction.operands.imm || '0');
            container.appendChild(document.createTextNode('('));
            addOperandInput(container, row, 'rs1', instruction.operands.rs1 || '');
            container.appendChild(document.createTextNode(')'));
        } else {
            addOperandInput(container, row, 'rs2', instruction.operands.rs2 || '');
            container.appendChild(document.createTextNode(', '));
            addOperandInput(container, row, 'imm', instruction.operands.imm || '0');
            container.appendChild(document.createTextNode('('));
            addOperandInput(container, row, 'rs1', instruction.operands.rs1 || '');
            container.appendChild(document.createTextNode(')'));
        }
    } else {
        operands.forEach((operand, idx) => {
            if (idx > 0) {
                container.appendChild(document.createTextNode(', '));
            }
            addOperandInput(container, row, operand, instruction.operands[operand] || '');
        });
    }
}

/**
 * Add a single operand input field to the container.
 *
 * @param {HTMLElement} container - The container to add the input to
 * @param {number} row - The row index
 * @param {string} operandName - The name of the operand (e.g., 'rd', 'rs1')
 * @param {string} value - The current value of the operand
 */
function addOperandInput(container, row, operandName, value) {
    const input = document.createElement('input');
    input.type = 'text';
    input.value = value;
    input.placeholder = operandName;
    input.dataset.operand = operandName;

    input.addEventListener('change', (e) => {
        if (!instructions[row]) {
            instructions[row] = { name: '', operands: {} };
        }
        instructions[row].operands[operandName] = e.target.value;
        saveInstruction(row);
    });

    container.appendChild(input);
}

/**
 * Update the instruction for a specific row when the instruction type changes.
 * Initializes default operand values based on the instruction format.
 *
 * @param {number} row - The row index
 * @param {string} instructionName - The name of the new instruction
 */
function updateInstruction(row, instructionName) {
    if (!instructionName) {
        delete instructions[row];
        saveInstruction(row);
        generateGrid();
        return;
    }

    const format = INSTRUCTION_FORMATS[instructionName];
    instructions[row] = {
        name: instructionName,
        operands: {}
    };

    format.operands.forEach(operand => {
        if (operand.startsWith('rs') || operand === 'rd') {
            instructions[row].operands[operand] = operand;
        } else if (operand === 'imm') {
            instructions[row].operands[operand] = '0';
        }
    });

    saveInstruction(row);
    generateGrid();
}

/**
 * Check all enabled rules and update the UI with violations.
 * Fetches violations from the server and highlights affected cells/rows.
 * @async
 */
async function checkRules() {
    const response = await fetch('/api/check-rules');
    const data = await response.json();
    currentViolations = data.violations || [];

    updateViolationUI();
}

/**
 * Update the UI to reflect current rule violations.
 * Highlights violated cells and rows, displays violation messages.
 */
/**
 * Update the UI to reflect the current set of rule violations.
 *
 * Handles both the original cell/row highlighting and the new
 * annotation-index highlighting introduced by WhenThenRule.
 *
 * After updating currentAnnotationHighlights, calls
 * renderBypassAnnotations() so the SVG overlay reflects the new state
 * without a separate explicit call.
 */
function updateViolationUI() {
    // ── Clear all existing highlights ────────────────────────────────
    document.querySelectorAll('.grid-cell').forEach(cell => {
        cell.classList.remove('violation');
    });
    document.querySelectorAll('.row-label').forEach(label => {
        label.classList.remove('violation');
    });

    const violationsSummary = document.getElementById('violations-summary');
    const violationsList    = document.getElementById('violations-list');

    if (currentViolations.length === 0) {
        violationsSummary.classList.remove('active');
        currentAnnotationHighlights = {};
        renderBypassAnnotations();
        return;
    }

    // ── Accumulate highlights ─────────────────────────────────────────
    const annotationHighlights = {};  // { index: hexColor }

    violationsSummary.classList.add('active');
    violationsList.innerHTML = '';

    currentViolations.forEach(violation => {
        // Grid cell highlights
        (violation.cells || []).forEach(cell => {
            const cellEl = document.querySelector(
                `.grid-cell[data-row="${cell.row}"][data-col="${cell.col}"]`
            );
            if (cellEl) cellEl.classList.add('violation');
        });

        // Row label highlights
        (violation.rows || []).forEach(rowNum => {
            const rowLabel = document.querySelector(
                `.row-label[data-row="${rowNum}"]`
            );
            if (rowLabel) rowLabel.classList.add('violation');
        });

        // Bypass annotation highlights (new)
        if (violation.annotation_indices && violation.highlight_color) {
            violation.annotation_indices.forEach(idx => {
                // Later violations overwrite earlier ones for the same index;
                // the last matching rule's color wins
                annotationHighlights[idx] = violation.highlight_color;
            });
        }

        // Violations list entry
        const item = document.createElement('div');
        item.textContent    = `• ${violation.message}`;
        item.style.marginLeft = '10px';
        item.style.marginTop  = '5px';
        violationsList.appendChild(item);
    });

    // ── Re-render SVG with new highlights ─────────────────────────────
    currentAnnotationHighlights = annotationHighlights;
    renderBypassAnnotations();
}
/**
 * Generate the scheduling grid and row labels with instruction editors.
 * Creates the complete grid UI including drag-and-drop handlers.
 */
function generateGrid() {
    const grid = document.getElementById('grid');
    const rowLabelsDiv = document.getElementById('row-labels');

    grid.innerHTML = '';
    rowLabelsDiv.innerHTML = '';
    grid.style.gridTemplateColumns = `repeat(${currentCols}, 50px)`;
    grid.style.gridTemplateRows = `repeat(${currentRows}, 50px)`;

    // Generate row labels with instruction editors
    for (let row = 0; row < currentRows; row++) {
        const labelDiv = document.createElement('div');
        labelDiv.className = 'row-label';
        labelDiv.dataset.row = row;

        const editor = createInstructionEditor(row);
        labelDiv.appendChild(editor);

        rowLabelsDiv.appendChild(labelDiv);
    }

    // Generate grid cells
    for (let row = 0; row < currentRows; row++) {
        for (let col = 0; col < currentCols; col++) {
            const cell = document.createElement('div');
            cell.className = 'grid-cell';
            cell.dataset.row = row;
            cell.dataset.col = col;

            // Check if there's a block in this cell
            const key = `${row},${col}`;
            if (gridData[key]) {
                const block = document.createElement('div');
                block.className = 'block';
                block.textContent = gridData[key];
                block.draggable = true;

                block.addEventListener('dragstart', (e) => {
                if (bypassAnnotationMode) {
                    e.preventDefault(); // Block dragging is disabled in pipeline mode
                    return;
                }
                e.dataTransfer.setData('blockType', gridData[key]);
                e.dataTransfer.setData('source', 'grid');
                e.dataTransfer.setData('sourceRow', row);
                e.dataTransfer.setData('sourceCol', col);
                block.classList.add('dragging');
                });

                block.addEventListener('dragend', (e) => {
                    block.classList.remove('dragging');
                });

                cell.appendChild(block);
            }

            // Drop event handlers
            cell.addEventListener('dragover', (e) => {
            if (bypassAnnotationMode) return; // Drops disabled in pipeline mode
            e.preventDefault();
            cell.classList.add('drag-over');
            });

            cell.addEventListener('dragleave', (e) => {
                cell.classList.remove('drag-over');
            });

            cell.addEventListener('drop', async (e) => {
                if (bypassAnnotationMode) return; // Drops disabled in pipeline mode
                e.preventDefault();
                cell.classList.remove('drag-over');

                const blockType = e.dataTransfer.getData('blockType');
                const source = e.dataTransfer.getData('source');

                if (source === 'grid') {
                    const sourceRow = parseInt(e.dataTransfer.getData('sourceRow'));
                    const sourceCol = parseInt(e.dataTransfer.getData('sourceCol'));
                    const sourceKey = `${sourceRow},${sourceCol}`;
                    delete gridData[sourceKey];
                    await saveBlock(sourceRow, sourceCol, null);
                }

                const key = `${row},${col}`;
                gridData[key] = blockType;
                await saveBlock(row, col, blockType);
                generateGrid();
                await checkRules();
            });

            // MODIFIED: Check bypass mode before the existing block-removal logic
            cell.addEventListener('contextmenu', async (e) => {
                e.preventDefault();

                // Cancel any in-progress drag first
                if (bypassDragging) {
                    bypassDragging = false;
                    bypassSource   = null;
                    document.querySelectorAll('.grid-cell.bypass-source')
                            .forEach(c => c.classList.remove('bypass-source'));
                }

                if (bypassAnnotationMode) {
                    await handleBypassCellRightClick(row, col);
                    return;
                }

                // Existing block-removal logic unchanged below
                const key = `${row},${col}`;
                if (gridData[key]) {
                    delete gridData[key];
                    await saveBlock(row, col, null);
                    generateGrid();
                    await checkRules();
                }
            });

            // mousedown starts a bypass annotation drag from this cell
            cell.addEventListener('mousedown', (e) => {
                if (!bypassAnnotationMode || e.button !== 0) return;
                e.preventDefault(); // Prevent browser text-selection during drag

                bypassSource   = {row, col};
                bypassDragging = true;

                cell.classList.add('bypass-source');
                renderBypassAnnotations();
            });

            // mouseup on this cell completes the annotation (source → this cell)
            cell.addEventListener('mouseup', (e) => {
                if (!bypassDragging || e.button !== 0) return;

                const source = bypassSource;

                // Reset drag state BEFORE any async work so the document-level
                // mouseup handler sees bypassDragging = false and does nothing
                bypassDragging = false;
                bypassSource   = null;
                document.querySelectorAll('.grid-cell.bypass-source')
                        .forEach(c => c.classList.remove('bypass-source'));

                const isSameCell = source && source.row === row && source.col === col;
                if (source && !isSameCell) {
                    saveBypassAnnotation(selectedBypassType, source, {row, col});
                } else {
                    renderBypassAnnotations(); // Redraw to remove the preview circle
                }
            });

            grid.appendChild(cell);
        }
    }

    // Check rules after grid generation
    renderBypassAnnotations(); // NEW: redraw SVG after grid is rebuilt
    checkRules();
}

// Slider event handlers
document.getElementById('rows-slider').addEventListener('input', (e) => {
    currentRows = parseInt(e.target.value);
    document.getElementById('rows-value').textContent = currentRows;
    generateGrid();
    saveResize();
});

document.getElementById('cols-slider').addEventListener('input', (e) => {
    currentCols = parseInt(e.target.value);
    document.getElementById('cols-value').textContent = currentCols;
    generateGrid();
    saveResize();
});

// Cancel a pipeline drag if the mouse button is released outside any grid cell
document.addEventListener('mouseup', (e) => {
    if (!bypassDragging || e.button !== 0) return;
    // If bypassDragging is still true here, the mouseup happened outside a cell
    bypassDragging = false;
    bypassSource   = null;
    document.querySelectorAll('.grid-cell.bypass-source')
             .forEach(c => c.classList.remove('bypass-source'));
    renderBypassAnnotations();
});

// Move the ghost circle to follow the cursor whenever annotation mode is active
document.addEventListener('mousemove', (e) => {
    if (!bypassAnnotationMode) return;
    const ghost = document.getElementById('ghost-circle');
    ghost.style.left = `${e.clientX}px`;
    ghost.style.top  = `${e.clientY}px`;
});

// Close the pipeline type dropdown when clicking anywhere outside the tool
document.addEventListener('click', (e) => {
    if (bypassMenuOpen && !e.target.closest('#bypass-tool')) {
        bypassMenuOpen = false;
        document.getElementById('bypass-type-list').classList.remove('visible');
    }
});

/**
 * Save a block placement to the server.
 *
 * @async
 * @param {number} row - Row index
 * @param {number} col - Column index
 * @param {string|null} blockType - Block type string or null to clear
 */
async function saveBlock(row, col, blockType) {
    await fetch('/api/block', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ row, col, block_type: blockType })
    });
}

/**
 * Save an instruction configuration to the server.
 *
 * @async
 * @param {number} row - Row index
 */
async function saveInstruction(row) {
    await fetch('/api/instruction', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            row,
            instruction: instructions[row] || { name: '', operands: {} }
        })
    });
    await checkRules();
}

/**
 * Save grid dimensions to the server.
 * @async
 */
async function saveResize() {
    await fetch('/api/resize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ rows: currentRows, cols: currentCols })
    });
}

/**
 * Load the complete scheduler state from the server.
 * Updates all client-side data structures and regenerates the UI.
 * @async
 */
async function loadStateFromServer() {
    const response = await fetch('/api/state');
    const state = await response.json();

    gridData = state.grid_data || {};
    instructions = state.instructions || {};
    currentRows = state.rows || 10;
    currentCols = state.cols || 10;
    bypassAnnotations = state.bypass_annotations || [];

    // Update sliders
    document.getElementById('rows-slider').value = currentRows;
    document.getElementById('cols-slider').value = currentCols;
    document.getElementById('rows-value').textContent = currentRows;
    document.getElementById('cols-value').textContent = currentCols;

    // Load rules info
    if (state.rules) {
        rulesInfo = state.rules;
        renderRulesPanel();
    }
    generateGrid();
}

/**
 * Save the current scheduler state to a JSON file.
 * Downloads the state as a JSON file for later restoration.
 */
function saveState() {

    const state = {
        grid_data:             gridData,
        instructions:          instructions,
        rows:                  currentRows,
        cols:                  currentCols,
        rules:                 rulesInfo,
        bypass_annotations:  bypassAnnotations   // NEW
    };

    const json = JSON.stringify(state, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'pipeline_state.json';
    a.click();
}

/**
 * Load scheduler state from a JSON file.
 * Prompts the user to select a file and restores the state.
 */
function loadState() {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'application/json';
    input.onchange = async (e) => {
        const file = e.target.files[0];
        const reader = new FileReader();
        reader.onload = async (event) => {
            const state = JSON.parse(event.target.result);
            gridData = state.grid_data || {};
            instructions = state.instructions || {};
            currentRows = state.rows || 10;
            currentCols = state.cols || 10;
            bypassAnnotations = state.bypass_annotations || [];

            // Update sliders
            document.getElementById('rows-slider').value = currentRows;
            document.getElementById('cols-slider').value = currentCols;
            document.getElementById('rows-value').textContent = currentRows;
            document.getElementById('cols-value').textContent = currentCols;

            // Update rules if provided
            if (state.rules) {
                rulesInfo = state.rules;
                renderRulesPanel();
            }

            generateGrid();

            // Send state to server
            await fetch('/api/state', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(state),
            });
        };
        reader.readAsText(file);
    };
    input.click();
}

/**
 * Clear the entire grid and all instructions.
 * Prompts for confirmation before clearing.
 * @async
 */
async function clearGrid() {
    if (confirm('Are you sure you want to clear the entire grid and all instructions?')) {
        gridData = {};
        instructions = {};
        generateGrid();

        bypassAnnotations = [];
        renderBypassAnnotations();

        await fetch('/api/state', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                grid_data: {},
                instructions: {},
                rows: currentRows,
                cols: currentCols,
                rules: rulesInfo,
                bypass_annotations: []
            })
        });

        // Explicitly check rules after clearing
        await checkRules();
    }
}

/**
 * Prompt the user to select a .yaml config file, upload it to the server,
 * and hot-reload all config-derived data structures without a page refresh.
 *
 * Validation model:
 *   - Errors are treated as blocking failures (import is aborted)
 *   - Warnings are non-blocking and do not prevent import
 *
 * On success (HTTP 200):
 *   - Reassigns BLOCK_TYPES, BYPASS_TYPES, INSTRUCTION_FORMATS,
 *     INSTRUCTION_NAMES from the server response
 *   - Updates the config name display
 *   - Displays warnings if present (non-blocking)
 *   - Calls initPalette(), initBypassTool(), loadRules(), generateGrid()
 *     to rebuild all affected UI sections
 *
 * On validation failure (HTTP 400):
 *   - Displays the list of validation error strings from the server
 *   - Aborts import without modifying application state
 *
 * On network or parsing failure:
 *   - Displays a generic error message and aborts import
 *
 * @async
 */
/**
 * Prompt the user to select a .yaml config file, upload it to the server,
 * and hot-reload all config-derived data structures without a page refresh.
 *
 * On success:
 *   - Reassigns BLOCK_TYPES, BYPASS_TYPES, INSTRUCTION_FORMATS,
 *     INSTRUCTION_NAMES from the server response
 *   - Updates the config name display
 *   - Displays any non-blocking warnings in the warnings banner
 *   - Calls initPalette(), initBypassTool(), loadRules(), generateGrid()
 *
 * On validation failure (HTTP 400):
 *   - Displays the list of blocking error strings in an alert
 *
 * @async
 */
async function importConfig() {
    const input  = document.createElement('input');
    input.type   = 'file';
    input.accept = '.yaml,.yml';

    input.onchange = async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        const formData = new FormData();
        formData.append('config', file);

        let response;
        try {
            response = await fetch('/api/config', {
                method: 'POST',
                body:   formData,
            });
        } catch (networkError) {
            console.error('importConfig: network error', networkError);
            alert('Failed to reach the server: ' + networkError.message);
            return;
        }

        const data = await response.json();

        if (!response.ok) {
            const errorText = (data.errors || ['Unknown error']).join('\n• ');
            alert(`Config validation failed:\n\n• ${errorText}`);
            return;
        }

        // ── Reassign config-derived globals ──────────────────────────
        BLOCK_TYPES         = data.block_types;
        BYPASS_TYPES        = data.bypass_types;
        INSTRUCTION_FORMATS = data.instruction_formats;
        INSTRUCTION_NAMES   = Object.keys(INSTRUCTION_FORMATS);

        // ── Config name display ───────────────────────────────────────
        const configNameEl = document.getElementById('config-name');
        if (configNameEl) {
            configNameEl.textContent = (data.meta && data.meta.name)
                ? data.meta.name
                : '';
        }

        // ── Non-blocking warnings ─────────────────────────────────────
        showConfigWarnings(data.warnings || []);

        // ── Rebuild affected UI sections ──────────────────────────────
        try { initPalette();    } catch (err) { console.error('initPalette failed:', err); }
        try { initBypassTool(); } catch (err) { console.error('initBypassTool failed:', err); }

        await loadRules();
        generateGrid();
    };

    input.click();
}

/**
 * Display non-blocking config warnings in the warnings banner.
 *
 * Hides the banner when the list is empty so it does not take up
 * space on a clean config load.  Each call fully replaces the previous
 * warning list.
 *
 * @param {string[]} warnings - Array of warning strings from the server
 */
function showConfigWarnings(warnings) {
    const banner = document.getElementById('config-warnings-banner');
    const list   = document.getElementById('config-warnings-list');
    if (!banner || !list) return;

    list.innerHTML = '';

    if (!warnings || warnings.length === 0) {
        banner.style.display = 'none';
        return;
    }

    warnings.forEach(w => {
        console.warn('Config warning:', w);
        const item       = document.createElement('li');
        item.textContent = w;
        item.style.marginBottom = '4px';
        list.appendChild(item);
    });

    banner.style.display = 'block';
}

// ─── Bypass tool initialisation ─────────────────────────────────────────

/**
 * Populate the bypass-type dropdown with one item per BYPASS_TYPES entry.
 * Each item shows a coloured dot and the type name + description.
 * Called once during page initialisation.
 */
function initBypassTool() {
    const typeList = document.getElementById('bypass-type-list');
    typeList.innerHTML = '';

    BYPASS_TYPES.forEach(type => {
        const item = document.createElement('div');
        item.className = 'bypass-type-item';

        const dot = document.createElement('div');
        dot.className = 'bypass-type-dot';
        dot.style.backgroundColor = type.color;

        const label = document.createElement('span');
        label.textContent = `${type.name} – ${type.description}`;

        item.appendChild(dot);
        item.appendChild(label);
        item.addEventListener('click', () => selectBypassType(type));
        typeList.appendChild(item);
    });
}

/**
 * Toggle the pipeline type dropdown, or exit annotation mode if already active.
 * Wired to the "Add Bypass" button's onclick attribute.
 */
function toggleBypassMenu() {
    if (bypassAnnotationMode) {
        exitPipelineMode();
        return;
    }
    bypassMenuOpen = !bypassMenuOpen;
    document.getElementById('bypass-type-list')
        .classList.toggle('visible', bypassMenuOpen);
}

/**
 * Enter bypass annotation mode for the given type.
 * Greys out the button, shows the ghost circle, and awaits the first cell click.
 *
 * @param {Object} type - A BYPASS_TYPES entry {name, color, description}
 */
/**
 * Enter bypass annotation mode for the given type.
 * Greys out the button, applies the bypass-mode CSS class to the grid
 * (which disables block interaction and enables cell hover), sets CSS
 * colour variables to match the type, and shows the ghost circle.
 *
 * @param {Object} type - A BYPASS_TYPES entry {name, color, description}
 */
function selectBypassType(type) {
    selectedBypassType   = type;
    bypassAnnotationMode = true;
    bypassSource         = null;
    bypassDragging       = false;
    bypassMenuOpen       = false;

    document.getElementById('bypass-type-list').classList.remove('visible');

    const btn = document.getElementById('bypass-btn');
    btn.classList.add('active');
    btn.textContent = `◉ ${type.name} (active – click to exit)`;

    // Add CSS class that drives hover highlight and pointer-event rules
    const grid = document.getElementById('grid');
    grid.classList.add('bypass-mode');

    // Set per-type CSS variables so hover colour matches the annotation colour
    const {r, g, b} = hexToRgb(type.color);
    grid.style.setProperty('--bypass-hover-bg',     `rgba(${r}, ${g}, ${b}, 0.15)`);
    grid.style.setProperty('--bypass-hover-border',  `rgba(${r}, ${g}, ${b}, 0.40)`);

    // Style the ghost circle to match
    const ghost = document.getElementById('ghost-circle');
    ghost.style.borderColor     = `rgba(${r}, ${g}, ${b}, 0.50)`;
    ghost.style.backgroundColor = `rgba(${r}, ${g}, ${b}, 0.12)`;
    ghost.style.display         = 'block';

    grid.style.cursor = 'crosshair';
}

/**
 * Exit bypass annotation mode and restore default UI state.
 * Removes bypass-mode CSS class (re-enables block dragging and hover),
 * hides the ghost circle, and resets all drag tracking variables.
 */
function exitPipelineMode() {
    bypassAnnotationMode = false;
    selectedBypassType   = null;
    bypassSource         = null;
    bypassDragging       = false;

    const btn = document.getElementById('bypass-btn');
    btn.classList.remove('active');
    btn.textContent = '✚ Add Bypass';

    const grid = document.getElementById('grid');
    grid.classList.remove('bypass-mode');
    grid.style.cursor = '';

    document.getElementById('ghost-circle').style.display = 'none';

    document.querySelectorAll('.grid-cell.bypass-source')
            .forEach(c => c.classList.remove('bypass-source'));

    renderBypassAnnotations();
}

// ─── Annotation placement ──────────────────────────────────────────────────

/**
 * Handle a right-click on a grid cell while annotation mode is active.
 * Removes every annotation that has this cell as its source or target.
 *
 * @async
 * @param {number} row
 * @param {number} col
 */
async function handleBypassCellRightClick(row, col) {
    await fetch('/api/bypass-annotations', {
        method: 'DELETE',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({row, col})
    });

    // Sync local state: drop annotations involving this cell
    bypassAnnotations = bypassAnnotations.filter(a =>
        !(a.source.row === row && a.source.col === col) &&
        !(a.target.row === row && a.target.col === col)
    );
    await checkRules();
    renderBypassAnnotations();
}

// ─── Persistence ──────────────────────────────────────────────────────────

/**
 * POST a new annotation to the server and add it to the local array.
 *
 * @async
 * @param {Object} type   - {name, color, description}
 * @param {Object} source - {row, col}
 * @param {Object} target - {row, col}
 */
async function saveBypassAnnotation(type, source, target) {
    const response = await fetch('/api/bypass-annotations', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({annotation_type: type.name, source, target})
    });
    const data = await response.json();
    if (data.success) {
        bypassAnnotations.push(data.annotation);
        renderBypassAnnotations();
        await checkRules();
    }
}

// ─── Rendering ────────────────────────────────────────────────────────────

/**
 * Return the pixel centre of a grid cell relative to the grid element.
 * Accounts for the 2 px border and 1 px gap in the CSS grid layout.
 *
 * @param {number} row
 * @param {number} col
 * @returns {{x: number, y: number}}
 */
function getCellCenter(row, col) {
    // border: 2px | gap: 1px | cell: 50px  →  centre = 2 + col*51 + 25
    return {
        x: 2 + col * 51 + 25,
        y: 2 + row * 51 + 25
    };
}

/**
 * Rebuild the SVG overlay from scratch.
 *
 * Draws:
 * - An arrowhead <marker> per distinct annotation colour
 * - A circle + arrow line for each saved annotation
 * - A dashed preview circle around the selected source cell (if any)
 */
function renderBypassAnnotations() {
    const svg  = document.getElementById('bypass-svg');
    const grid = document.getElementById('grid');
    if (!svg || !grid) return;

    svg.setAttribute('width',  grid.offsetWidth  || 0);
    svg.setAttribute('height', grid.offsetHeight || 0);
    svg.innerHTML = '';

    // ── Arrowhead markers (one per colour) ──
    const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
    svg.appendChild(defs);

    const colorsNeeded = new Set(bypassAnnotations.map(a => a.color));
    if (selectedBypassType) colorsNeeded.add(selectedBypassType.color);

    colorsNeeded.forEach(color => {
        const id     = `arrow-${color.replace('#', '')}`;
        const marker = document.createElementNS('http://www.w3.org/2000/svg', 'marker');
        marker.setAttribute('id',           id);
        marker.setAttribute('markerWidth',  '8');
        marker.setAttribute('markerHeight', '6');
        marker.setAttribute('refX',         '7');
        marker.setAttribute('refY',         '3');
        marker.setAttribute('orient',       'auto');

        const poly = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
        poly.setAttribute('points', '0 0, 8 3, 0 6');
        poly.setAttribute('fill',   color);
        marker.appendChild(poly);
        defs.appendChild(marker);
    });

    // ── Draw saved annotations ──
    bypassAnnotations.forEach((a, idx) => {
        const overrideColor  = currentAnnotationHighlights[idx] || null;
        const isHighlighted  = idx in currentAnnotationHighlights;
        drawAnnotation(
            svg,
            a.source,
            a.target,
            a.color,
            isHighlighted,
            overrideColor
        );
    });

    // ── Draw dashed preview circle around source (if awaiting target click) ──
    if (bypassSource && selectedBypassType) {
        const c      = getCellCenter(bypassSource.row, bypassSource.col);
        const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        circle.setAttribute('cx',           c.x);
        circle.setAttribute('cy',           c.y);
        circle.setAttribute('r',            28);
        circle.setAttribute('fill',         'none');
        circle.setAttribute('stroke',       selectedBypassType.color);
        circle.setAttribute('stroke-width', '2');
        circle.setAttribute('stroke-dasharray', '5,3');
        svg.appendChild(circle);
    }
    renderAnnotationList(); // Keep the list panel in sync with the SVG
}

/**
 * Rebuild the annotation list panel from the current bypassAnnotations array.
 *
 * Each item shows a colored dot, the annotation type, source cell
 * (row + instruction name if available + column), and target cell in the
 * same format, plus a delete button.
 *
 * Called automatically at the end of renderBypassAnnotations().
 */
function renderAnnotationList() {
    const list = document.getElementById('annotation-list');
    if (!list) return;

    list.innerHTML = '';

    if (bypassAnnotations.length === 0) {
        const empty = document.createElement('div');
        empty.className = 'annotation-panel-empty';
        empty.textContent = 'No annotations yet.';
        list.appendChild(empty);
        return;
    }

    bypassAnnotations.forEach(annotation => {
        // ── Resolve human-readable labels for each endpoint ──
        const srcInstr = instructions[annotation.source.row];
        const tgtInstr = instructions[annotation.target.row];

        const srcName = (srcInstr && srcInstr.name)
            ? `Row ${annotation.source.row} (${srcInstr.name}), col ${annotation.source.col}`
            : `Row ${annotation.source.row}, col ${annotation.source.col}`;

        const tgtName = (tgtInstr && tgtInstr.name)
            ? `Row ${annotation.target.row} (${tgtInstr.name}), col ${annotation.target.col}`
            : `Row ${annotation.target.row}, col ${annotation.target.col}`;

        // ── Build DOM ──
        const item = document.createElement('div');
        item.className = 'annotation-item';
        item.style.borderLeftColor = annotation.color;

        const dot = document.createElement('div');
        dot.className = 'annotation-dot';
        dot.style.backgroundColor = annotation.color;

        const info = document.createElement('div');
        info.className = 'annotation-info';
        info.innerHTML =
            `<strong>${annotation.annotation_type}</strong><br>` +
            `${srcName}<br>` +
            `&rarr; ${tgtName}`;

        const deleteBtn = document.createElement('button');
        deleteBtn.className = 'annotation-delete';
        deleteBtn.textContent = '✕';
        deleteBtn.title = 'Remove this annotation';
        deleteBtn.addEventListener('click', () => deleteSpecificAnnotation(annotation));

        item.appendChild(dot);
        item.appendChild(info);
        item.appendChild(deleteBtn);
        list.appendChild(item);
    });
}

/**
 * Delete one specific annotation identified by its source, target, and type.
 * Sends a DELETE request with all three fields so the server can match
 * exactly one annotation even if multiple share a source or target cell.
 *
 * @async
 * @param {Object} annotation - An entry from bypassAnnotations
 */
async function deleteSpecificAnnotation(annotation) {
    await fetch('/api/bypass-annotations', {
        method: 'DELETE',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            source:          annotation.source,
            target:          annotation.target,
            annotation_type: annotation.annotation_type
        })

    });

    // Sync client-side array
    bypassAnnotations = bypassAnnotations.filter(a =>
        !(a.source.row      === annotation.source.row  &&
          a.source.col      === annotation.source.col  &&
          a.target.row      === annotation.target.row  &&
          a.target.col      === annotation.target.col  &&
          a.annotation_type === annotation.annotation_type)
    );

    await checkRules();
    renderBypassAnnotations()
}

/**
 * Draw a single annotation: a solid circle around the source cell and
 * a line with an arrowhead pointing to the target cell.
 *
 * The line starts at the edge of the source circle (r = 28 px) and ends
 * just inside the boundary of the target cell.
 *
 * @param {SVGElement} svg
 * @param {{row:number, col:number}} source
 * @param {{row:number, col:number}} target
 * @param {string}                   color   - Hex colour string
 */
/**
 * Draw one bypass annotation onto the SVG overlay.
 *
 * Renders a circle around the source cell and a line with an arrowhead
 * pointing to the target cell.  When isHighlighted is true, the stroke
 * is thickened and a glow filter is applied to draw attention to the
 * violation.
 *
 * @param {SVGElement} svg
 * @param {{row:number, col:number}} source
 * @param {{row:number, col:number}} target
 * @param {string}  color         - Hex color string for normal drawing
 * @param {boolean} isHighlighted - True when a rule violation applies
 * @param {string}  [highlightColor] - Override color when highlighted;
 *                                     defaults to color if not supplied
 */
function drawAnnotation(svg, source, target, color,
                        isHighlighted = false,
                        highlightColor = null) {
    const CIRCLE_R  = 28;
    const CELL_HALF = 25;
    const drawColor = (isHighlighted && highlightColor) ? highlightColor : color;
    const markerId  = `arrow-${drawColor.replace('#', '')}`;
    const strokeW   = isHighlighted ? '3' : '2';

    const src = getCellCenter(source.row, source.col);
    const tgt = getCellCenter(target.row, target.col);
    const dx  = tgt.x - src.x;
    const dy  = tgt.y - src.y;
    const len = Math.sqrt(dx * dx + dy * dy);
    if (len === 0) return;

    const nx = dx / len;
    const ny = dy / len;

    // ── Circle around source ──────────────────────────────────────────
    const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    circle.setAttribute('cx',           src.x);
    circle.setAttribute('cy',           src.y);
    circle.setAttribute('r',            CIRCLE_R);
    circle.setAttribute('fill',         'none');
    circle.setAttribute('stroke',       drawColor);
    circle.setAttribute('stroke-width', strokeW);
    if (isHighlighted) {
        circle.setAttribute('stroke-dasharray', '');  // solid when highlighted
        circle.setAttribute('filter',
            `drop-shadow(0 0 4px ${drawColor})`);
    }
    svg.appendChild(circle);

    // ── Arrow line ────────────────────────────────────────────────────
    const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    line.setAttribute('x1',           src.x + nx * CIRCLE_R);
    line.setAttribute('y1',           src.y + ny * CIRCLE_R);
    line.setAttribute('x2',           tgt.x - nx * (CELL_HALF - 2));
    line.setAttribute('y2',           tgt.y - ny * (CELL_HALF - 2));
    line.setAttribute('stroke',       drawColor);
    line.setAttribute('stroke-width', strokeW);
    line.setAttribute('marker-end',   `url(#${markerId})`);
    if (isHighlighted) {
        line.setAttribute('filter',
            `drop-shadow(0 0 4px ${drawColor})`);
    }
    svg.appendChild(line);
}

// Initialize
initPalette();
initBypassTool();
loadRules().then(() => {
    loadStateFromServer();
});
