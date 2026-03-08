const BLOCK_TYPES = ['F', 'D', 'i', 'I', 'Y0', 'Y1', 'Y2', 'Y3', 'W', 'r', 'C', 'X'];
const INSTRUCTION_FORMATS = window.INSTRUCTION_FORMATS
const INSTRUCTION_NAMES = Object.keys(INSTRUCTION_FORMATS);

let currentRows = 10;
let currentCols = 10;
let gridData = {};
let instructions = {};
let currentViolations = [];
let rulesInfo = [];
let currentPipelineCount = window.PIPELINE_COUNT;

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
                e.preventDefault();
                cell.classList.add('drag-over');
            });

            cell.addEventListener('dragleave', (e) => {
                cell.classList.remove('drag-over');
            });

            cell.addEventListener('drop', async (e) => {
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

            // Right-click to remove block
            cell.addEventListener('contextmenu', async (e) => {
                e.preventDefault();
                const key = `${row},${col}`;
                if (gridData[key]) {
                    delete gridData[key];
                    await saveBlock(row, col, null);
                    generateGrid();
                    await checkRules();
                }
            });

            grid.appendChild(cell);
        }
    }

    // Check rules after grid generation
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

    generateGrid();
}

/**
 * Save the current scheduler state to a JSON file.
 * Downloads the state as a JSON file for later restoration.
 */
function saveState() {
    const state = {
        grid_data: gridData,
        instructions: instructions,
        rows: currentRows,
        cols: currentCols,
        pipeline_count: currentPipelineCount,
        rules: rulesInfo
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
                body: JSON.stringify(state)
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

        await fetch('/api/state', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                grid_data: {},
                instructions: {},
                rows: currentRows,
                cols: currentCols,
                pipeline_count: currentPipelineCount,
                rules: rulesInfo
            })
        });

        // Explicitly check rules after clearing
        await checkRules();
    }
}

// Initialize
initPalette();
loadRules().then(() => {
    loadStateFromServer();
});