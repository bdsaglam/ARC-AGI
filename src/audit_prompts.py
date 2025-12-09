from src.grid import grid_to_string, grid_to_csv_rows
from src.audit_templates_logic import (
    PROMPT_LOGIC_SYSTEM_ROLE,
    PROMPT_LOGIC_INSTRUCTIONS
)
from src.audit_templates_consistency import (
    PROMPT_CONSISTENCY_SYSTEM_ROLE,
    PROMPT_CONSISTENCY_TASK_CONTEXT,
    PROMPT_CONSISTENCY_INSTRUCTIONS_FORMAT,
    PROMPT_CONSISTENCY_OUTPUT_FORMAT
)

def build_logic_prompt(train_examples, test_input, candidates_list):
    logic_parts = []
    logic_parts.append(PROMPT_LOGIC_SYSTEM_ROLE)
    logic_parts.append("\n<INPUT_DATA>")
    
    logic_parts.append("1. {SOLVED_EXAMPLES}:")
    for i, example in enumerate(train_examples):
        logic_parts.append(f"<EXAMPLE_{i+1}>")
        logic_parts.append("<INPUT>")
        logic_parts.append(grid_to_string(example.input))
        logic_parts.append("</INPUT>")
        logic_parts.append("<OUTPUT>")
        logic_parts.append(grid_to_string(example.output))
        logic_parts.append("</OUTPUT>")
        logic_parts.append(f"</EXAMPLE_{i+1}>")
    
    logic_parts.append("\n2. {TEST_INPUT}:")
    if test_input:
        logic_parts.append(grid_to_string(test_input))
    else:
        logic_parts.append("(No Test Input)")

    logic_parts.append("\n3. {CANDIDATES}:")
    for cand in candidates_list:
        c_id = cand['id']
        logic_parts.append(f"<CANDIDATE {c_id}>")
        logic_parts.append("<PROPOSED_SOLUTION>")
        logic_parts.append(grid_to_string(cand['grid']))
        logic_parts.append("</PROPOSED_SOLUTION>")
        for j, model_id in enumerate(cand['models']):
            alias = chr(65 + j)
            logic_parts.append(f'<REASONING_MODEL_{alias} model_id="{model_id}">')
            reasoning = cand['reasoning'].get(model_id, "(Reasoning not found)")
            logic_parts.append(reasoning)
            logic_parts.append(f"</REASONING_MODEL_{alias}>")
        logic_parts.append(f"</CANDIDATE {c_id}>")

    logic_parts.append("</INPUT_DATA>\n")
    logic_parts.append(PROMPT_LOGIC_INSTRUCTIONS)
    return "\n".join(logic_parts)

def build_consistency_prompt(train_examples, test_input, candidates_list):
    cons_parts = []
    cons_parts.append(PROMPT_CONSISTENCY_SYSTEM_ROLE)
    cons_parts.append(PROMPT_CONSISTENCY_TASK_CONTEXT)
    cons_parts.append("\n<PROBLEM>")
    
    for i, ex in enumerate(train_examples):
        cons_parts.append(f'  <TRAIN_EXAMPLE index="{i+1}">')
        cons_parts.append("    <INPUT_GRID>")
        cons_parts.append(grid_to_csv_rows(ex.input))
        cons_parts.append("    </INPUT_GRID>")
        cons_parts.append("    <OUTPUT_GRID>")
        cons_parts.append(grid_to_csv_rows(ex.output))
        cons_parts.append("    </OUTPUT_GRID>")
        cons_parts.append("  </TRAIN_EXAMPLE>")
        
    if test_input:
        cons_parts.append("  <TEST_INPUT>")
        cons_parts.append("    <INPUT_GRID>")
        cons_parts.append(grid_to_csv_rows(test_input))
        cons_parts.append("    </INPUT_GRID>")
        cons_parts.append("  </TEST_INPUT>")
        
    cons_parts.append("</PROBLEM>\n")
    
    cons_parts.append("<CANDIDATES>")
    for cand in candidates_list:
        c_id = cand['id']
        cons_parts.append(f'  <CANDIDATE id="{c_id}">')
        for j, model_id in enumerate(cand['models']):
            alias = chr(65 + j)
            cons_parts.append(f'    <ANSWER id="{alias}" model_id="{model_id}">')
            cons_parts.append(f'      <EXPLANATION>')
            reasoning = cand['reasoning'].get(model_id, "(Reasoning not found)")
            cons_parts.append(reasoning)
            cons_parts.append(f'      </EXPLANATION>')
            cons_parts.append(f'      <OUTPUT_GRID>')
            cons_parts.append(grid_to_csv_rows(cand['grid']))
            cons_parts.append(f'      </OUTPUT_GRID>')
            cons_parts.append(f'    </ANSWER>')
        cons_parts.append(f'  </CANDIDATE>')
    cons_parts.append("</CANDIDATES>\n")
    
    cons_parts.append(PROMPT_CONSISTENCY_INSTRUCTIONS_FORMAT)
    cons_parts.append(PROMPT_CONSISTENCY_OUTPUT_FORMAT)
    return "\n".join(cons_parts)