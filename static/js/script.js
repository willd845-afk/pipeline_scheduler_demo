// ── Flask-injected data ────────────────────────────────────────────────────
// Read from window.FLASK_DATA which was set by the inline script in index.html.
// Keeping these separate from FLASK_DATA avoids redeclaration errors and lets
// the rest of the file use the same short names as before.
const INSTRUCTION_FORMATS  = window.FLASK_DATA.instructionFormats;
const INSTRUCTION_NAMES    = Object.keys(INSTRUCTION_FORMATS);
const PIPELINE_TYPES       = window.FLASK_DATA.pipelineTypes;
let   currentPipelineCount = window.FLASK_DATA.pipelineCount;

// ── Static constants ───────────────────────────────────────────────────────
const BLOCK_TYPES = ['F', 'D', 'i', 'I', 'Y0', 'Y1', 'Y2', 'Y3', 'W', 'r', 'C', 'X'];

let currentRows = 10;
let currentCols = 10;
let gridData = {};
let instructions = {};
let currentViolations = [];
let rulesInfo = [];

// Annotation mode state
let pipelineAnnotationMode = false;   // true while placing annotations
let selectedPipelineType  = null;     // the PipelineType object currently selected
let pipelineSource        = null;     // {row, col} of the first click; null until set
let pipelineAnnotations   = [];       // all annotations loaded from / saved to server
let pipelineMenuOpen      = false;    // whether the type dropdown is visible
let pipelineDragging = false; // true while the user is holding mousedown on a source cell

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
        block.dataset.blockType
        block.dataset.blockType = blockType;

        block.addEventListener('dragstart', (e) => {
            e.dataTransfer.setData('blockType', blockType);
            e.dataTransfer.setData('source', 'palette');
            block.classList.add('dragging');
        });

        block.addEventListener('dragend', (e) => {
            block.classList.remove('dragging');
        });

        // Exit annotation mode when the user starts dragging a palette block
        block.addEventListener('dragstart', (e) => {
            if (pipelineAnnotationMode) exitPipelineMode(); // NEW
            e.dataTransfer.setData('blockType', blockType);
            e.dataTransfer.setData('source', 'palette');
            block.classList.add('dragging');
        });

        // Also exit if the user merely clicks a palette block (without dragging)
        block.addEventListener('click', () => {          // NEW
            if (pipelineAnnotationMode) exitPipelineMode();
        });

        palette.appendChild(block);
    });
}

function initPipelinePalette() {
    const pipelineButton = document.createElement("BUTTON");
    pipelineButton.className = 'pipeline-button';
    pipelineButton.addEventListener('click', (e) => {

        /*
        * Clicking brings up a list of pipeline types
        * Swap to new state where clicking adds (lmb) or removes (rmb) pipelines
        * Have a transparent circle around the mouse
        * Greyed out button while enabled
        * Mode is disabled when palette block is clicked
        */

    })


    pipelineButton
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
 * Update the pipeline count based on radio button selection.
 * Sends the new count to the server and triggers rule re-checking.
 * @async
 */
async function updatePipelineCount() {
    const selected = document.querySelector('input[name="pipelines"]:checked');
    if (selected) {
        currentPipelineCount = parseInt(selected.value);
        await fetch('/api/pipeline-count', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ pipeline_count: currentPipelineCount })
        });
        await checkRules();
    }
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
function updateViolationUI() {
    // Clear all violation classes
    document.querySelectorAll('.grid-cell').forEach(cell => {
        cell.classList.remove('violation');
    });
    document.querySelectorAll('.row-label').forEach(label => {
        label.classList.remove('violation');
    });

    const violationsSummary = document.getElementById('violations-summary');
    const violationsList = document.getElementById('violations-list');

    if (currentViolations.length === 0) {
        violationsSummary.classList.remove('active');
        return;
    }

    // Show violations summary
    violationsSummary.classList.add('active');
    violationsList.innerHTML = '';

    currentViolations.forEach(violation => {
        // Highlight affected cells
        violation.cells.forEach(cell => {
            const cellElement = document.querySelector(
                `.grid-cell[data-row="${cell.row}"][data-col="${cell.col}"]`
            );
            if (cellElement) {
                cellElement.classList.add('violation');
            }
        });

        // Highlight affected rows
        violation.rows.forEach(rowNum => {
            const rowLabel = document.querySelector(
                `.row-label[data-row="${rowNum}"]`
            );
            if (rowLabel) {
                rowLabel.classList.add('violation');
            }
        });

        // Add violation to list
        const violationItem = document.createElement('div');
        violationItem.textContent = `• ${violation.message}`;
        violationItem.style.marginLeft = '10px';
        violationItem.style.marginTop = '5px';
        violationsList.appendChild(violationItem);
    });
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
                if (pipelineAnnotationMode) {
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
            if (pipelineAnnotationMode) return; // Drops disabled in pipeline mode
            e.preventDefault();
            cell.classList.add('drag-over');
            });

            cell.addEventListener('dragleave', (e) => {
                cell.classList.remove('drag-over');
            });

            cell.addEventListener('drop', async (e) => {
                if (pipelineAnnotationMode) return; // Drops disabled in pipeline mode
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

            // MODIFIED: Check pipeline mode before the existing block-removal logic
            cell.addEventListener('contextmenu', async (e) => {
                e.preventDefault();

                // Cancel any in-progress drag first
                if (pipelineDragging) {
                    pipelineDragging = false;
                    pipelineSource   = null;
                    document.querySelectorAll('.grid-cell.pipeline-source')
                            .forEach(c => c.classList.remove('pipeline-source'));
                }

                if (pipelineAnnotationMode) {
                    await handlePipelineCellRightClick(row, col);
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

            // mousedown starts a pipeline annotation drag from this cell
            cell.addEventListener('mousedown', (e) => {
                if (!pipelineAnnotationMode || e.button !== 0) return;
                e.preventDefault(); // Prevent browser text-selection during drag

                pipelineSource   = {row, col};
                pipelineDragging = true;

                cell.classList.add('pipeline-source');
                renderPipelineAnnotations();
            });

            // mouseup on this cell completes the annotation (source → this cell)
            cell.addEventListener('mouseup', (e) => {
                if (!pipelineDragging || e.button !== 0) return;

                const source = pipelineSource;

                // Reset drag state BEFORE any async work so the document-level
                // mouseup handler sees pipelineDragging = false and does nothing
                pipelineDragging = false;
                pipelineSource   = null;
                document.querySelectorAll('.grid-cell.pipeline-source')
                        .forEach(c => c.classList.remove('pipeline-source'));

                const isSameCell = source && source.row === row && source.col === col;
                if (source && !isSameCell) {
                    savePipelineAnnotation(selectedPipelineType, source, {row, col});
                } else {
                    renderPipelineAnnotations(); // Redraw to remove the preview circle
                }
            });

            grid.appendChild(cell);
        }
    }

    // Check rules after grid generation
    renderPipelineAnnotations(); // NEW: redraw SVG after grid is rebuilt
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
    if (!pipelineDragging || e.button !== 0) return;
    // If pipelineDragging is still true here, the mouseup happened outside a cell
    pipelineDragging = false;
    pipelineSource   = null;
    document.querySelectorAll('.grid-cell.pipeline-source')
             .forEach(c => c.classList.remove('pipeline-source'));
    renderPipelineAnnotations();
});

// Move the ghost circle to follow the cursor whenever annotation mode is active
document.addEventListener('mousemove', (e) => {
    if (!pipelineAnnotationMode) return;
    const ghost = document.getElementById('ghost-circle');
    ghost.style.left = `${e.clientX}px`;
    ghost.style.top  = `${e.clientY}px`;
});

// Close the pipeline type dropdown when clicking anywhere outside the tool
document.addEventListener('click', (e) => {
    if (pipelineMenuOpen && !e.target.closest('#pipeline-tool')) {
        pipelineMenuOpen = false;
        document.getElementById('pipeline-type-list').classList.remove('visible');
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
    currentPipelineCount = state.pipeline_count || 1;

    // Update sliders
    document.getElementById('rows-slider').value = currentRows;
    document.getElementById('cols-slider').value = currentCols;
    document.getElementById('rows-value').textContent = currentRows;
    document.getElementById('cols-value').textContent = currentCols;

    // Update pipeline radio buttons
    const pipelineRadio = document.querySelector(`input[name="pipelines"][value="${currentPipelineCount}"]`);
    if (pipelineRadio) {
        pipelineRadio.checked = true;
    }

    // Load rules info
    if (state.rules) {
        rulesInfo = state.rules;
        renderRulesPanel();
    }

    pipelineAnnotations = state.pipeline_annotations || [];
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
        pipeline_count:        currentPipelineCount,
        rules:                 rulesInfo,
        pipeline_annotations:  pipelineAnnotations   // NEW
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
            currentPipelineCount = state.pipeline_count || 1;

            pipelineAnnotations = state.pipeline_annotations || [];

            // Update sliders
            document.getElementById('rows-slider').value = currentRows;
            document.getElementById('cols-slider').value = currentCols;
            document.getElementById('rows-value').textContent = currentRows;
            document.getElementById('cols-value').textContent = currentCols;

            // Update pipeline radio buttons
            const pipelineRadio = document.querySelector(`input[name="pipelines"][value="${currentPipelineCount}"]`);
            if (pipelineRadio) {
                pipelineRadio.checked = true;
            }

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
                pipeline_annotations: pipelineAnnotations
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

        pipelineAnnotations = [];
        renderPipelineAnnotations();

        await fetch('/api/state', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                grid_data: {},
                instructions: {},
                rows: currentRows,
                cols: currentCols,
                pipeline_count: currentPipelineCount,
                rules: rulesInfo,
                pipeline_annotations: []
            })
        });

        // Explicitly check rules after clearing
        await checkRules();
    }
}

// ─── Pipeline tool initialisation ─────────────────────────────────────────

/**
 * Populate the pipeline-type dropdown with one item per PIPELINE_TYPES entry.
 * Each item shows a coloured dot and the type name + description.
 * Called once during page initialisation.
 */
function initPipelineTool() {
    const typeList = document.getElementById('pipeline-type-list');
    typeList.innerHTML = '';

    PIPELINE_TYPES.forEach(type => {
        const item = document.createElement('div');
        item.className = 'pipeline-type-item';

        const dot = document.createElement('div');
        dot.className = 'pipeline-type-dot';
        dot.style.backgroundColor = type.color;

        const label = document.createElement('span');
        label.textContent = `${type.name} – ${type.description}`;

        item.appendChild(dot);
        item.appendChild(label);
        item.addEventListener('click', () => selectPipelineType(type));
        typeList.appendChild(item);
    });
}

/**
 * Toggle the pipeline type dropdown, or exit annotation mode if already active.
 * Wired to the "Add Pipeline" button's onclick attribute.
 */
function togglePipelineMenu() {
    if (pipelineAnnotationMode) {
        exitPipelineMode();
        return;
    }
    pipelineMenuOpen = !pipelineMenuOpen;
    document.getElementById('pipeline-type-list')
        .classList.toggle('visible', pipelineMenuOpen);
}

/**
 * Enter pipeline annotation mode for the given type.
 * Greys out the button, shows the ghost circle, and awaits the first cell click.
 *
 * @param {Object} type - A PIPELINE_TYPES entry {name, color, description}
 */
/**
 * Enter pipeline annotation mode for the given type.
 * Greys out the button, applies the pipeline-mode CSS class to the grid
 * (which disables block interaction and enables cell hover), sets CSS
 * colour variables to match the type, and shows the ghost circle.
 *
 * @param {Object} type - A PIPELINE_TYPES entry {name, color, description}
 */
function selectPipelineType(type) {
    selectedPipelineType   = type;
    pipelineAnnotationMode = true;
    pipelineSource         = null;
    pipelineDragging       = false;
    pipelineMenuOpen       = false;

    document.getElementById('pipeline-type-list').classList.remove('visible');

    const btn = document.getElementById('pipeline-btn');
    btn.classList.add('active');
    btn.textContent = `◉ ${type.name} (active – click to exit)`;

    // Add CSS class that drives hover highlight and pointer-event rules
    const grid = document.getElementById('grid');
    grid.classList.add('pipeline-mode');

    // Set per-type CSS variables so hover colour matches the annotation colour
    const {r, g, b} = hexToRgb(type.color);
    grid.style.setProperty('--pipeline-hover-bg',     `rgba(${r}, ${g}, ${b}, 0.15)`);
    grid.style.setProperty('--pipeline-hover-border',  `rgba(${r}, ${g}, ${b}, 0.40)`);

    // Style the ghost circle to match
    const ghost = document.getElementById('ghost-circle');
    ghost.style.borderColor     = `rgba(${r}, ${g}, ${b}, 0.50)`;
    ghost.style.backgroundColor = `rgba(${r}, ${g}, ${b}, 0.12)`;
    ghost.style.display         = 'block';

    grid.style.cursor = 'crosshair';
}

/**
 * Exit pipeline annotation mode and restore default UI state.
 * Removes pipeline-mode CSS class (re-enables block dragging and hover),
 * hides the ghost circle, and resets all drag tracking variables.
 */
function exitPipelineMode() {
    pipelineAnnotationMode = false;
    selectedPipelineType   = null;
    pipelineSource         = null;
    pipelineDragging       = false;

    const btn = document.getElementById('pipeline-btn');
    btn.classList.remove('active');
    btn.textContent = '✚ Add Pipeline';

    const grid = document.getElementById('grid');
    grid.classList.remove('pipeline-mode');
    grid.style.cursor = '';

    document.getElementById('ghost-circle').style.display = 'none';

    document.querySelectorAll('.grid-cell.pipeline-source')
            .forEach(c => c.classList.remove('pipeline-source'));

    renderPipelineAnnotations();
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
async function handlePipelineCellRightClick(row, col) {
    await fetch('/api/pipeline-annotations', {
        method: 'DELETE',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({row, col})
    });

    // Sync local state: drop annotations involving this cell
    pipelineAnnotations = pipelineAnnotations.filter(a =>
        !(a.source.row === row && a.source.col === col) &&
        !(a.target.row === row && a.target.col === col)
    );

    renderPipelineAnnotations();
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
async function savePipelineAnnotation(type, source, target) {
    const response = await fetch('/api/pipeline-annotations', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({annotation_type: type.name, source, target})
    });
    const data = await response.json();
    if (data.success) {
        pipelineAnnotations.push(data.annotation);
        renderPipelineAnnotations();
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
function renderPipelineAnnotations() {
    const svg  = document.getElementById('pipeline-svg');
    const grid = document.getElementById('grid');
    if (!svg || !grid) return;

    svg.setAttribute('width',  grid.offsetWidth  || 0);
    svg.setAttribute('height', grid.offsetHeight || 0);
    svg.innerHTML = '';

    // ── Arrowhead markers (one per colour) ──
    const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
    svg.appendChild(defs);

    const colorsNeeded = new Set(pipelineAnnotations.map(a => a.color));
    if (selectedPipelineType) colorsNeeded.add(selectedPipelineType.color);

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
    pipelineAnnotations.forEach(a => drawAnnotation(svg, a.source, a.target, a.color));

    // ── Draw dashed preview circle around source (if awaiting target click) ──
    if (pipelineSource && selectedPipelineType) {
        const c      = getCellCenter(pipelineSource.row, pipelineSource.col);
        const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        circle.setAttribute('cx',           c.x);
        circle.setAttribute('cy',           c.y);
        circle.setAttribute('r',            28);
        circle.setAttribute('fill',         'none');
        circle.setAttribute('stroke',       selectedPipelineType.color);
        circle.setAttribute('stroke-width', '2');
        circle.setAttribute('stroke-dasharray', '5,3');
        svg.appendChild(circle);
    }
    renderAnnotationList(); // Keep the list panel in sync with the SVG
}

/**
 * Rebuild the annotation list panel from the current pipelineAnnotations array.
 *
 * Each item shows a coloured dot, the annotation type, source cell
 * (row + instruction name if available + column), and target cell in the
 * same format, plus a delete button.
 *
 * Called automatically at the end of renderPipelineAnnotations().
 */
function renderAnnotationList() {
    const list = document.getElementById('annotation-list');
    if (!list) return;

    list.innerHTML = '';

    if (pipelineAnnotations.length === 0) {
        const empty = document.createElement('div');
        empty.className = 'annotation-panel-empty';
        empty.textContent = 'No annotations yet.';
        list.appendChild(empty);
        return;
    }

    pipelineAnnotations.forEach(annotation => {
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
 * @param {Object} annotation - An entry from pipelineAnnotations
 */
async function deleteSpecificAnnotation(annotation) {
    await fetch('/api/pipeline-annotations', {
        method: 'DELETE',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            source:          annotation.source,
            target:          annotation.target,
            annotation_type: annotation.annotation_type
        })
    });

    // Sync client-side array
    pipelineAnnotations = pipelineAnnotations.filter(a =>
        !(a.source.row      === annotation.source.row  &&
          a.source.col      === annotation.source.col  &&
          a.target.row      === annotation.target.row  &&
          a.target.col      === annotation.target.col  &&
          a.annotation_type === annotation.annotation_type)
    );

    renderPipelineAnnotations(); // Also calls renderAnnotationList
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
function drawAnnotation(svg, source, target, color) {
    const CIRCLE_R  = 28;
    const CELL_HALF = 25;
    const markerId  = `arrow-${color.replace('#', '')}`;

    const src = getCellCenter(source.row, source.col);
    const tgt = getCellCenter(target.row, target.col);
    const dx  = tgt.x - src.x;
    const dy  = tgt.y - src.y;
    const len = Math.sqrt(dx * dx + dy * dy);
    if (len === 0) return;

    const nx = dx / len;
    const ny = dy / len;

    // Circle around source
    const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    circle.setAttribute('cx',           src.x);
    circle.setAttribute('cy',           src.y);
    circle.setAttribute('r',            CIRCLE_R);
    circle.setAttribute('fill',         'none');
    circle.setAttribute('stroke',       color);
    circle.setAttribute('stroke-width', '2');
    svg.appendChild(circle);

    // Arrow from source circle edge to target cell edge
    const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    line.setAttribute('x1',           src.x + nx * CIRCLE_R);
    line.setAttribute('y1',           src.y + ny * CIRCLE_R);
    line.setAttribute('x2',           tgt.x - nx * (CELL_HALF - 2));
    line.setAttribute('y2',           tgt.y - ny * (CELL_HALF - 2));
    line.setAttribute('stroke',       color);
    line.setAttribute('stroke-width', '2');
    line.setAttribute('marker-end',   `url(#${markerId})`);
    svg.appendChild(line);
}

// Initialize
initPalette();
initPipelineTool();
loadRules().then(() => {
    loadStateFromServer();
});