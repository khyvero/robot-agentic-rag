# generate L1 architecture diagrams showing system components and connections

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from pathlib import Path

COLOR_WHITE = '#FFFFFF'
COLOR_BLACK = '#000000'
COLOR_GRAY = '#888888'

output_dir = Path('reports')

# zero-shot L1 architecture
fig1, ax1 = plt.subplots(figsize=(12, 8))
ax1.set_xlim(0, 12)
ax1.set_ylim(0, 8)
ax1.axis('off')
ax1.set_title('Zero-Shot System Architecture', fontsize=16, fontweight='bold', pad=20)

# user interface layer
ui_box = Rectangle((1, 6.5), 2.5, 1, edgecolor=COLOR_BLACK, facecolor=COLOR_WHITE, linewidth=2)
ax1.add_patch(ui_box)
ax1.text(2.25, 7, 'User Interface', ha='center', va='center', fontsize=11, fontweight='bold')

# system prompt component
prompt_box = Rectangle((5, 6.5), 2.5, 1, edgecolor=COLOR_BLACK, facecolor=COLOR_WHITE, linewidth=2)
ax1.add_patch(prompt_box)
ax1.text(6.25, 7, 'System Prompt\nDefinitions', ha='center', va='center', fontsize=10, fontweight='bold')

# llm engine
llm_box = Rectangle((8.5, 6.5), 2.5, 1, edgecolor=COLOR_BLACK, facecolor=COLOR_WHITE, linewidth=2.5)
ax1.add_patch(llm_box)
ax1.text(9.75, 7, 'Llama 3.8b\nLLM Engine', ha='center', va='center', fontsize=11, fontweight='bold')

# json processor
json_box = Rectangle((5, 4), 2.5, 1, edgecolor=COLOR_BLACK, facecolor=COLOR_WHITE, linewidth=2)
ax1.add_patch(json_box)
ax1.text(6.25, 4.5, 'JSON Processor\n& Validator', ha='center', va='center', fontsize=10, fontweight='bold')

# robot executor
robot_box = Rectangle((5, 1), 2.5, 1, edgecolor=COLOR_BLACK, facecolor=COLOR_WHITE, linewidth=2)
ax1.add_patch(robot_box)
ax1.text(6.25, 1.5, 'Robot Executor', ha='center', va='center', fontsize=11, fontweight='bold')

# connections
arrow1 = FancyArrowPatch((3.5, 7), (5, 7), arrowstyle='<->', mutation_scale=20, linewidth=2, color=COLOR_BLACK)
ax1.add_patch(arrow1)

arrow2 = FancyArrowPatch((7.5, 7), (8.5, 7), arrowstyle='->', mutation_scale=20, linewidth=2, color=COLOR_BLACK)
ax1.add_patch(arrow2)
ax1.text(8, 7.3, 'prompt', ha='center', fontsize=8)

arrow3 = FancyArrowPatch((9.75, 6.5), (6.25, 5), arrowstyle='->', mutation_scale=20, linewidth=2, color=COLOR_BLACK)
ax1.add_patch(arrow3)
ax1.text(8.5, 5.7, 'raw JSON', ha='center', fontsize=8)

arrow4 = FancyArrowPatch((6.25, 4), (6.25, 2), arrowstyle='->', mutation_scale=20, linewidth=2, color=COLOR_BLACK)
ax1.add_patch(arrow4)
ax1.text(6.6, 3, 'valid JSON', ha='left', fontsize=8)

# ollama runtime box
ollama_box = Rectangle((8, 5.5), 4, 2.5, edgecolor=COLOR_GRAY, facecolor='none', linewidth=1.5, linestyle='--')
ax1.add_patch(ollama_box)
ax1.text(10, 5.2, 'Ollama Runtime', ha='center', fontsize=9, color=COLOR_GRAY, style='italic')

plt.tight_layout()
plt.savefig(output_dir / 'zero_shot_architecture.png', dpi=300, bbox_inches='tight', facecolor='white')
print("saved zero_shot_architecture.png")
plt.close()

# single-rag L1 architecture
fig2, ax2 = plt.subplots(figsize=(12, 10))
ax2.set_xlim(0, 12)
ax2.set_ylim(0, 10)
ax2.axis('off')
ax2.set_title('Single-RAG System Architecture', fontsize=16, fontweight='bold', pad=20)

# user interface
ui_box2 = Rectangle((1, 8.5), 2, 1, edgecolor=COLOR_BLACK, facecolor=COLOR_WHITE, linewidth=2)
ax2.add_patch(ui_box2)
ax2.text(2, 9, 'User Interface', ha='center', va='center', fontsize=11, fontweight='bold')

# embedding service
emb_box = Rectangle((4.5, 8.5), 2.5, 1, edgecolor=COLOR_BLACK, facecolor=COLOR_WHITE, linewidth=2)
ax2.add_patch(emb_box)
ax2.text(5.75, 9, 'Sentence\nTransformer', ha='center', va='center', fontsize=10, fontweight='bold')

# unified knowledge base
kb_box = Rectangle((8.5, 8.5), 2.5, 1, edgecolor=COLOR_BLACK, facecolor=COLOR_WHITE, linewidth=2.5)
ax2.add_patch(kb_box)
ax2.text(9.75, 9, 'Unified KB\n(ChromaDB)', ha='center', va='center', fontsize=10, fontweight='bold')

# retrieval service
ret_box = Rectangle((8.5, 6.5), 2.5, 1, edgecolor=COLOR_BLACK, facecolor=COLOR_WHITE, linewidth=2)
ax2.add_patch(ret_box)
ax2.text(9.75, 7, 'Retrieval Service\n(cosine sim)', ha='center', va='center', fontsize=10, fontweight='bold')

# context merger
merge_box = Rectangle((5.5, 6.5), 2, 1, edgecolor=COLOR_BLACK, facecolor=COLOR_WHITE, linewidth=2)
ax2.add_patch(merge_box)
ax2.text(6.5, 7, 'Context\nMerger', ha='center', va='center', fontsize=10, fontweight='bold')

# llm engine
llm_box2 = Rectangle((3, 4.5), 2.5, 1, edgecolor=COLOR_BLACK, facecolor=COLOR_WHITE, linewidth=2.5)
ax2.add_patch(llm_box2)
ax2.text(4.25, 5, 'Llama 3.8b\nLLM Engine', ha='center', va='center', fontsize=11, fontweight='bold')

# json processor
json_box2 = Rectangle((7, 4.5), 2.5, 1, edgecolor=COLOR_BLACK, facecolor=COLOR_WHITE, linewidth=2)
ax2.add_patch(json_box2)
ax2.text(8.25, 5, 'JSON Processor\n& Validator', ha='center', va='center', fontsize=10, fontweight='bold')

# robot executor
robot_box2 = Rectangle((5, 2), 2.5, 1, edgecolor=COLOR_BLACK, facecolor=COLOR_WHITE, linewidth=2)
ax2.add_patch(robot_box2)
ax2.text(6.25, 2.5, 'Robot Executor', ha='center', va='center', fontsize=11, fontweight='bold')

# knowledge storage (data layer)
data_box1 = Rectangle((8.2, 7.8), 1.1, 0.6, edgecolor=COLOR_GRAY, facecolor=COLOR_WHITE, linewidth=1)
ax2.add_patch(data_box1)
ax2.text(8.75, 8.1, 'Decl\nRecipes', ha='center', va='center', fontsize=7)

data_box2 = Rectangle((9.4, 7.8), 1.1, 0.6, edgecolor=COLOR_GRAY, facecolor=COLOR_WHITE, linewidth=1)
ax2.add_patch(data_box2)
ax2.text(9.95, 8.1, 'Proc\nAPIs', ha='center', va='center', fontsize=7)

# connections
arrow2_1 = FancyArrowPatch((3, 9), (4.5, 9), arrowstyle='->', mutation_scale=20, linewidth=2, color=COLOR_BLACK)
ax2.add_patch(arrow2_1)

arrow2_2 = FancyArrowPatch((7, 9), (8.5, 9), arrowstyle='->', mutation_scale=20, linewidth=2, color=COLOR_BLACK)
ax2.add_patch(arrow2_2)
ax2.text(7.7, 9.3, 'query', ha='center', fontsize=8)

arrow2_3 = FancyArrowPatch((9.75, 8.5), (9.75, 7.5), arrowstyle='->', mutation_scale=20, linewidth=2, color=COLOR_BLACK)
ax2.add_patch(arrow2_3)

arrow2_4 = FancyArrowPatch((8.5, 7), (7.5, 7), arrowstyle='->', mutation_scale=20, linewidth=2, color=COLOR_BLACK)
ax2.add_patch(arrow2_4)
ax2.text(8, 7.3, 'context', ha='center', fontsize=8)

arrow2_5 = FancyArrowPatch((2, 8.5), (2, 7.5), arrowstyle='->', mutation_scale=20, linewidth=1.5, color=COLOR_GRAY, linestyle='--')
ax2.add_patch(arrow2_5)
ax2.text(2.5, 8, 'original\nprompt', ha='left', fontsize=7, color=COLOR_GRAY)

arrow2_6 = FancyArrowPatch((2, 7.5), (5.5, 7), arrowstyle='->', mutation_scale=15, linewidth=1.5, color=COLOR_GRAY, linestyle='--')
ax2.add_patch(arrow2_6)

arrow2_7 = FancyArrowPatch((6.5, 6.5), (4.25, 5.5), arrowstyle='->', mutation_scale=20, linewidth=2, color=COLOR_BLACK)
ax2.add_patch(arrow2_7)
ax2.text(5.5, 6.2, 'augmented\nprompt', ha='center', fontsize=8)

arrow2_8 = FancyArrowPatch((5.5, 5), (7, 5), arrowstyle='->', mutation_scale=20, linewidth=2, color=COLOR_BLACK)
ax2.add_patch(arrow2_8)

arrow2_9 = FancyArrowPatch((7.5, 4.5), (6.25, 3), arrowstyle='->', mutation_scale=20, linewidth=2, color=COLOR_BLACK)
ax2.add_patch(arrow2_9)

# ollama runtime
ollama_box2 = Rectangle((2.5, 4), 3.5, 2, edgecolor=COLOR_GRAY, facecolor='none', linewidth=1.5, linestyle='--')
ax2.add_patch(ollama_box2)
ax2.text(4.25, 3.7, 'Ollama Runtime', ha='center', fontsize=9, color=COLOR_GRAY, style='italic')

plt.tight_layout()
plt.savefig(output_dir / 'single_rag_architecture.png', dpi=300, bbox_inches='tight', facecolor='white')
print("saved single_rag_architecture.png")
plt.close()

# dual-rag L1 architecture
fig3, ax3 = plt.subplots(figsize=(16, 12))
ax3.set_xlim(0, 16)
ax3.set_ylim(0, 12)
ax3.axis('off')
ax3.set_title('Dual-RAG System Architecture', fontsize=16, fontweight='bold', pad=20)

# user interface
ui_box3 = Rectangle((7, 10.5), 2, 1, edgecolor=COLOR_BLACK, facecolor=COLOR_WHITE, linewidth=2)
ax3.add_patch(ui_box3)
ax3.text(8, 11, 'User Interface', ha='center', va='center', fontsize=11, fontweight='bold')

# fsm controller
fsm_box = Rectangle((10.5, 10.5), 2, 1, edgecolor=COLOR_BLACK, facecolor=COLOR_WHITE, linewidth=2)
ax3.add_patch(fsm_box)
ax3.text(11.5, 11, 'FSM Controller\n(State Machine)', ha='center', va='center', fontsize=10, fontweight='bold')

# intent router agent
router_box = Rectangle((7, 8.5), 2, 1, edgecolor=COLOR_BLACK, facecolor=COLOR_WHITE, linewidth=2)
ax3.add_patch(router_box)
ax3.text(8, 9, 'Intent Router\nAgent', ha='center', va='center', fontsize=10, fontweight='bold')

# match evaluator (sub-component)
eval_box = Rectangle((10, 8.5), 2, 1, edgecolor=COLOR_BLACK, facecolor=COLOR_WHITE, linewidth=1.5)
ax3.add_patch(eval_box)
ax3.text(11, 9, 'Match Quality\nEvaluator (LLM)', ha='center', va='center', fontsize=9, fontweight='bold')

# declarative kb
decl_kb_box = Rectangle((1, 6.5), 2.5, 1.5, edgecolor=COLOR_BLACK, facecolor=COLOR_WHITE, linewidth=2.5)
ax3.add_patch(decl_kb_box)
ax3.text(2.25, 7.5, 'Declarative KB', ha='center', va='center', fontsize=10, fontweight='bold')
ax3.text(2.25, 7.1, '(ChromaDB)', ha='center', va='center', fontsize=8)

# declarative sub-collections
stage1_box = Rectangle((1.1, 6.7), 1, 0.5, edgecolor=COLOR_GRAY, facecolor=COLOR_WHITE, linewidth=1)
ax3.add_patch(stage1_box)
ax3.text(1.6, 6.95, 'Stage 1\nKeywords', ha='center', va='center', fontsize=7)

stage2_box = Rectangle((2.3, 6.7), 1, 0.5, edgecolor=COLOR_GRAY, facecolor=COLOR_WHITE, linewidth=1)
ax3.add_patch(stage2_box)
ax3.text(2.8, 6.95, 'Stage 2\nSemantic', ha='center', va='center', fontsize=7)

# procedural kb
proc_kb_box = Rectangle((4.5, 6.5), 2.5, 1.5, edgecolor=COLOR_BLACK, facecolor=COLOR_WHITE, linewidth=2.5)
ax3.add_patch(proc_kb_box)
ax3.text(5.75, 7.5, 'Procedural KB', ha='center', va='center', fontsize=10, fontweight='bold')
ax3.text(5.75, 7.1, '(ChromaDB)', ha='center', va='center', fontsize=8)

# procedural data
proc_data_box = Rectangle((4.6, 6.7), 2.3, 0.5, edgecolor=COLOR_GRAY, facecolor=COLOR_WHITE, linewidth=1)
ax3.add_patch(proc_data_box)
ax3.text(5.75, 6.95, 'API Specifications', ha='center', va='center', fontsize=7)

# embedding service
emb_box3 = Rectangle((8.5, 6.5), 2, 1, edgecolor=COLOR_BLACK, facecolor=COLOR_WHITE, linewidth=2)
ax3.add_patch(emb_box3)
ax3.text(9.5, 7, 'Sentence\nTransformer', ha='center', va='center', fontsize=10, fontweight='bold')

# tier 2 agents
mod_box = Rectangle((1, 4.5), 2, 0.8, edgecolor=COLOR_BLACK, facecolor=COLOR_WHITE, linewidth=1.5)
ax3.add_patch(mod_box)
ax3.text(2, 4.9, 'Modification\nDetector', ha='center', va='center', fontsize=9, fontweight='bold')

action_box = Rectangle((3.5, 4.5), 2, 0.8, edgecolor=COLOR_BLACK, facecolor=COLOR_WHITE, linewidth=1.5)
ax3.add_patch(action_box)
ax3.text(4.5, 4.9, 'Action\nExtractor', ha='center', va='center', fontsize=9, fontweight='bold')

proc_ret_box = Rectangle((6, 4.5), 2.2, 0.8, edgecolor=COLOR_BLACK, facecolor=COLOR_WHITE, linewidth=1.5)
ax3.add_patch(proc_ret_box)
ax3.text(7.1, 4.9, 'Procedural\nRetrieval Service', ha='center', va='center', fontsize=9, fontweight='bold')

# plan generator
plan_gen_box = Rectangle((9, 4.5), 2.5, 0.8, edgecolor=COLOR_BLACK, facecolor=COLOR_WHITE, linewidth=2.5)
ax3.add_patch(plan_gen_box)
ax3.text(10.25, 4.9, 'Plan Generator\n(LLM)', ha='center', va='center', fontsize=10, fontweight='bold')

# conversation agent
conv_box = Rectangle((12, 4.5), 2.5, 0.8, edgecolor=COLOR_BLACK, facecolor=COLOR_WHITE, linewidth=2)
ax3.add_patch(conv_box)
ax3.text(13.25, 4.9, 'Conversation\nAgent', ha='center', va='center', fontsize=10, fontweight='bold')

# json processor
json_box3 = Rectangle((6.5, 2.5), 2.5, 1, edgecolor=COLOR_BLACK, facecolor=COLOR_WHITE, linewidth=2)
ax3.add_patch(json_box3)
ax3.text(7.75, 3, 'JSON Processor\n& Validator', ha='center', va='center', fontsize=10, fontweight='bold')

# robot executor
robot_box3 = Rectangle((6.5, 0.5), 2.5, 1, edgecolor=COLOR_BLACK, facecolor=COLOR_WHITE, linewidth=2)
ax3.add_patch(robot_box3)
ax3.text(7.75, 1, 'Robot Executor', ha='center', va='center', fontsize=11, fontweight='bold')

# connections - user to router
arrow3_1 = FancyArrowPatch((8, 10.5), (8, 9.5), arrowstyle='->', mutation_scale=20, linewidth=2, color=COLOR_BLACK)
ax3.add_patch(arrow3_1)

# fsm to router
arrow3_1b = FancyArrowPatch((10.5, 11), (9, 11), arrowstyle='<->', mutation_scale=15, linewidth=1.5, color=COLOR_GRAY, linestyle='--')
ax3.add_patch(arrow3_1b)

# router to evaluator
arrow3_2 = FancyArrowPatch((9, 9), (10, 9), arrowstyle='<->', mutation_scale=20, linewidth=1.5, color=COLOR_BLACK)
ax3.add_patch(arrow3_2)

# router to declarative kb
arrow3_3 = FancyArrowPatch((7, 9), (3.5, 7.5), arrowstyle='<->', mutation_scale=20, linewidth=2, color=COLOR_BLACK)
ax3.add_patch(arrow3_3)
ax3.text(5, 8.5, 'query recipes', ha='center', fontsize=8)

# router to embedding service
arrow3_4 = FancyArrowPatch((8.5, 9), (9.5, 8), arrowstyle='->', mutation_scale=15, linewidth=1.5, color=COLOR_GRAY, linestyle='--')
ax3.add_patch(arrow3_4)

# router to tier 2 agents
arrow3_5a = FancyArrowPatch((7.5, 8.5), (2, 5.3), arrowstyle='->', mutation_scale=15, linewidth=1.5, color=COLOR_BLACK)
ax3.add_patch(arrow3_5a)

arrow3_5b = FancyArrowPatch((7.8, 8.5), (4.5, 5.3), arrowstyle='->', mutation_scale=15, linewidth=1.5, color=COLOR_BLACK)
ax3.add_patch(arrow3_5b)

# declarative kb to agents
arrow3_6 = FancyArrowPatch((2.25, 6.5), (3, 5.3), arrowstyle='->', mutation_scale=15, linewidth=1.5, color=COLOR_GRAY, linestyle='--')
ax3.add_patch(arrow3_6)

# procedural kb to retrieval service
arrow3_7 = FancyArrowPatch((5.75, 6.5), (7.1, 5.3), arrowstyle='<->', mutation_scale=20, linewidth=2, color=COLOR_BLACK)
ax3.add_patch(arrow3_7)

# action extractor to proc retrieval
arrow3_8 = FancyArrowPatch((5.5, 4.9), (6, 4.9), arrowstyle='->', mutation_scale=20, linewidth=1.5, color=COLOR_BLACK)
ax3.add_patch(arrow3_8)

# agents to plan generator
arrow3_9a = FancyArrowPatch((3, 4.9), (9, 4.9), arrowstyle='->', mutation_scale=15, linewidth=1.5, color=COLOR_GRAY, linestyle='--')
ax3.add_patch(arrow3_9a)

arrow3_9b = FancyArrowPatch((8.2, 4.9), (9, 4.9), arrowstyle='->', mutation_scale=20, linewidth=1.5, color=COLOR_BLACK)
ax3.add_patch(arrow3_9b)

# plan gen to conversation agent
arrow3_10 = FancyArrowPatch((11.5, 4.9), (12, 4.9), arrowstyle='->', mutation_scale=20, linewidth=2, color=COLOR_BLACK)
ax3.add_patch(arrow3_10)

# conversation to json processor
arrow3_11 = FancyArrowPatch((13.25, 4.5), (7.75, 3.5), arrowstyle='->', mutation_scale=20, linewidth=2, color=COLOR_BLACK)
ax3.add_patch(arrow3_11)

# json to robot
arrow3_12 = FancyArrowPatch((7.75, 2.5), (7.75, 1.5), arrowstyle='->', mutation_scale=20, linewidth=2, color=COLOR_BLACK)
ax3.add_patch(arrow3_12)

# conversation to fsm (state updates)
arrow3_13 = FancyArrowPatch((13.25, 5.3), (11.5, 10.5), arrowstyle='->', mutation_scale=15, linewidth=1.5, color=COLOR_GRAY, linestyle='--')
ax3.add_patch(arrow3_13)
ax3.text(12.5, 8, 'state\nupdates', ha='center', fontsize=7, color=COLOR_GRAY)

# conversation to user (feedback loop)
arrow3_14 = FancyArrowPatch((14.5, 4.9), (14.5, 10.5), arrowstyle='->', mutation_scale=15, linewidth=1.5, color=COLOR_GRAY, linestyle='--')
ax3.add_patch(arrow3_14)
ax3.text(15, 7.5, 'review\nfeedback', ha='left', fontsize=7, color=COLOR_GRAY, rotation=90, va='center')

arrow3_14b = FancyArrowPatch((14.5, 10.5), (9, 10.5), arrowstyle='->', mutation_scale=15, linewidth=1.5, color=COLOR_GRAY, linestyle='--')
ax3.add_patch(arrow3_14b)

# ollama runtime boxes
ollama_box3a = Rectangle((9.5, 8.2), 3, 1.6, edgecolor=COLOR_GRAY, facecolor='none', linewidth=1.5, linestyle='--')
ax3.add_patch(ollama_box3a)
ax3.text(11, 8, 'Ollama', ha='center', fontsize=8, color=COLOR_GRAY, style='italic')

ollama_box3b = Rectangle((8.5, 4), 3, 1.6, edgecolor=COLOR_GRAY, facecolor='none', linewidth=1.5, linestyle='--')
ax3.add_patch(ollama_box3b)
ax3.text(10, 3.7, 'Ollama', ha='center', fontsize=8, color=COLOR_GRAY, style='italic')

# tier labels
tier2_label = Rectangle((0.5, 4), 7.8, 1.6, edgecolor=COLOR_GRAY, facecolor='none', linewidth=1, linestyle=':')
ax3.add_patch(tier2_label)
ax3.text(0.3, 5.3, 'Tier 2 Pipeline', ha='left', fontsize=9, color=COLOR_GRAY, style='italic', rotation=90, va='center')

plt.tight_layout()
plt.savefig(output_dir / 'dual_rag_architecture.png', dpi=300, bbox_inches='tight', facecolor='white')
print("saved dual_rag_architecture.png")
plt.close()

print("\ngenerated 3 L1 architecture diagrams with white boxes")
