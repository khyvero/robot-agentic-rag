# Robotic Arm Control with Multi-Agent LLM Architecture

### Description
This is a python-based system for controlling a simulated robotic arm in RoboDK using natural language commands processed by a Large Language Model (Llama 3 via Ollama). The system features a **Multi-Agent Architecture** with specialized agents for intent routing, action extraction, and plan generation, translating user commands into structured robotic tasks.

### Key Features

#### Multi-Agent Architecture
*   **Intent Router Agent**: Routes user prompts to appropriate processing tiers based on semantic similarity
*   **Action Extractor Agent**: Identifies required robot actions using LLM with dynamic action keyword matching
*   **Plan Generator Agent**: Generates and validates JSON mission plans with dual-mode support (refinement vs. generation)
*   **Conversation Agent**: Handles multi-turn dialogue for ambiguity resolution and plan review
*   **Procedural Retrieval Service**: Intelligent API retrieval using multi-query strategies

#### Planning Strategies
*   **Dual-RAG Strategy**: Three-tier architecture
    - **Tier 1**: Intent routing with semantic similarity (exact match, ambiguous, novel task)
    - **Tier 2**: Recipe refinement with declarative + procedural context
    - **Tier 3**: Novel task generation with human-in-the-loop plan review
*   **Single-RAG Strategy**: Unified knowledge retrieval
*   **Zero-Shot Strategy**: Direct LLM generation without RAG

#### Robotic Capabilities
*   **10 Atomic Actions**: pick, place, place_free_spot, pour, shake, swirl, move_home, ensure_gripper_empty, place_in_area, wait
*   **Smart Placement**: Automatic free spot detection to avoid collisions
*   **Object Manipulation**: Pick, place, pour, shake, swirl operations
*   **Safety Features**: Gripper state management, collision avoidance

### Technology Stack
*   **Python 3.8+**
*   **RoboDK**: For robot simulation and control
*   **Ollama**: For running the Llama 3 LLM locally
*   **ChromaDB**: For vector-based knowledge retrieval
*   **Pydantic**: For type-safe data validation

---

## Installation

### Prerequisites
1.  **RoboDK**: Download and install [RoboDK](https://robodk.com/download)
2.  **Ollama**: Download and install [Ollama](https://ollama.com/)

### Setup Steps

1.  **Clone the repository**:
    ```sh
    git clone <repository_url>
    cd robot-agentic-rag
    ```

2.  **Set up a virtual environment (Recommended)**:
    ```sh
    python -m venv .venv
    # Activate on Windows:
    .venv\Scripts\activate
    # Activate on macOS/Linux:
    source .venv/bin/activate
    ```

3.  **Install dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

4.  **Setup Ollama**:

    After installing Ollama, start the server and download the model:

    *   **Start the Server**:
        *   **Terminal (macOS/Linux)**:
            ```sh
            ollama serve
            ```
            *Keep this terminal open while using the tool.*
        *   **Desktop App (macOS/Windows)**: Launch the **Ollama** application

    *   **Pull the Model**:
        ```sh
        ollama pull llama3:8b
        ```

---

## Usage

### Basic Usage

1.  **Start RoboDK**:
    - Open the application and load your station
    - Ensure the **UR10e** robotic arm and **RobotiQ 2F-85 Gripper (Mechanism)** are present
    - Ensure the Tool (TCP) is added as child of gripper
    - Ensure objects like **test_tube_blood**, **beaker_water**, **biohazard_bin**, etc. are present

2.  **Start Ollama**: Ensure the Ollama app is running

3.  **Run the main script**:
    ```sh
    python main.py
    ```

4.  **Enter commands**:

    Example commands:
    *   `"Pick up test_tube_blood"`
    *   `"Move beaker_water to biohazard_bin"`
    *   `"Pour test_tube_blood into beaker_water"`
    *   `"Shake the test_tube and place it in the bin"`
    *   `"Mix the blood and DNA"` (triggers novel task generation)

### Advanced Usage

#### Strategy Selection

The system supports three planning strategies:

```python
from core.orchestrator import Orchestrator

# Use Dual-RAG Strategy (default, recommended)
orchestrator = Orchestrator(strategy_name="dual_rag")

# Use Single-RAG Strategy
orchestrator = Orchestrator(strategy_name="single_rag")

# Use Zero-Shot Strategy
orchestrator = Orchestrator(strategy_name="zero_shot")
```

#### Configuration

Customize behavior in `config/config.py`:

```python
# Multi-Agent Configuration
INTENT_ROUTER_THRESHOLD_EXACT = 1.1      # Distance < 1.1 = Exact match
INTENT_ROUTER_THRESHOLD_AMBIGUOUS = 1.6  # 1.1 <= Distance < 1.6 = Ambiguous

# Procedural Retrieval Configuration
PROCEDURAL_MIN_RESULTS = 3   # Minimum APIs to retrieve

# Action Extraction Configuration
ACTION_EXTRACTION_ENABLED = True  # Feature flag for action extraction
```

---

## Architecture Overview

### Multi-Agent System

```
┌──────────────────────────────────────────────────┐
│         DualRAGStrategy (Orchestrator)           │
│                                                  │
│  - FSM state management                          │
│  - Delegates to specialized agents               │
│  - Assembles final RobotMission                  │
└────┬──────┬──────┬──────┬──────┬────────────────┘
     │      │      │      │      │
     ▼      ▼      ▼      ▼      ▼
┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
│Intent  │ │Action  │ │  Plan  │ │Conversa│ │Procedur│
│Router  │ │Extract.│ │Generat.│ │tion    │ │Retriev.│
└────────┘ └────────┘ └────────┘ └────────┘ └────────┘
```

### Three-Tier Processing

1. **Tier 1 (Intent Routing)**:
   - Semantic similarity search in declarative knowledge
   - Routes to: Exact Match, Ambiguous, Novel Task, or Non-Task

2. **Tier 2 (Recipe Refinement)**:
   - Uses retrieved recipe + procedural APIs
   - Action extraction for targeted retrieval
   - Generates plan with LLM

3. **Tier 3 (Novel Task Generation)**:
   - Generates new plans from scratch
   - Human-in-the-loop review and modification
   - Iterative plan refinement

### Data Flow

```
User Prompt
    ↓
Intent Router → [EXACT_MATCH | AMBIGUOUS | NOVEL_TASK | NOT_TASK]
    ↓
Action Extractor → ["pick", "place", "pour"]
    ↓
Procedural Retrieval → API Documentation
    ↓
Plan Generator → JSON Mission Plan
    ↓
[Tier 2: Execute] or [Tier 3: Review → Modify → Confirm]
    ↓
RobotMission (tasks + settings)
```

---

## Project Structure

```
robot-agentic-rag/
├── core/
│   ├── agents/                    # Multi-agent architecture
│   │   ├── intent_router.py      # Routes user prompts
│   │   ├── action_extractor.py   # Extracts required actions
│   │   ├── plan_generator.py     # Generates JSON plans
│   │   ├── conversation_agent.py # Handles dialogue
│   │   └── types.py               # Pydantic models for I/O
│   ├── services/
│   │   └── procedural_retrieval.py # Intelligent API retrieval
│   ├── strategies/
│   │   ├── dual_rag.py           # Dual-RAG strategy (multi-agent)
│   │   ├── single_rag.py         # Single-RAG strategy
│   │   └── zero_shot.py          # Zero-shot strategy
│   ├── knowledge_base.py         # ChromaDB interface
│   ├── orchestrator.py           # Main orchestration layer
│   ├── robot_control.py          # RoboDK interface
│   └── mission_executor.py       # Task execution
├── config/
│   ├── config.py                 # Configuration parameters
│   └── prompts.py                # LLM prompt templates
├── data/
│   └── knowledge/
│       ├── procedural_api.json   # Robot actions with keywords
│       └── declarative_tasks.json # Task recipes
├── reports/
│   └── ARCHITECTURE_ENHANCEMENT_PLAN.md  # Detailed architecture docs
├── tests/                        # Unit and integration tests
├── main.py                       # Entry point
└── README.md                     # This file
```

---

## Testing

### Robot Control Unit Tests
```sh
python -m unittest tests.test_robot_control_unittest
```

### Robot Control Integration Tests
```sh
python tests/test_robot_control_integration.py
```

### Agent Unit Tests (to be implemented)
```sh
python -m pytest tests/test_intent_router.py
python -m pytest tests/test_action_extractor.py
python -m pytest tests/test_plan_generator.py
python -m pytest tests/test_conversation_agent.py
python -m pytest tests/test_procedural_retrieval.py
```

### Integration Tests (to be implemented)
```sh
python -m pytest tests/test_dual_rag_integration.py
```

---

## Knowledge Base

### Procedural APIs (10 actions)

The system supports 10 robot actions with comprehensive keyword mappings:

| Action | Keywords | Description |
|--------|----------|-------------|
| `pick` | pick, pick up, grab, take, get, hold, lift, grasp | Picks up an object by name |
| `place` | place, put, put down, drop, set down, position, set | Places held object at destination |
| `place_free_spot` | place freely, find spot, avoid stacking, place separately | Places object in free spot |
| `pour` | pour, empty, transfer liquid, dispense | Pours liquid from held object |
| `shake` | shake, agitate, mix by shaking | Shakes held object |
| `swirl` | swirl, rotate, circular motion, mix gently | Swirls held object in circle |
| `move_home` | move home, return, reset position, go to start | Moves to home position |
| `ensure_gripper_empty` | release, drop, ensure empty, clear gripper | Ensures gripper is empty |
| `place_in_area` | place in area, place in region, constrained placement | Places in specified area |
| `wait` | wait, pause, delay, hold position | Waits for specified time |

### Declarative Tasks

The system includes pre-defined task recipes for common laboratory operations:
- Blood sample handling
- DNA extraction procedures
- Chemical mixing protocols
- Waste disposal workflows

To add new tasks, edit `data/knowledge/declarative_tasks.json`.

---

## Configuration Parameters

### LLM Configuration
```python
LLM_MODEL = "llama3:8b"
LLM_TEMPERATURE = 0.0  # Deterministic for reproducibility
LLM_TIMEOUT = 120  # Seconds
```

### Multi-Agent Configuration
```python
INTENT_ROUTER_THRESHOLD_EXACT = 1.1      # Distance < 1.1 = Exact match
INTENT_ROUTER_THRESHOLD_AMBIGUOUS = 1.6  # 1.1 <= Distance < 1.6 = Ambiguous
PROCEDURAL_MIN_RESULTS = 3               # Minimum APIs to retrieve
ACTION_EXTRACTION_ENABLED = True         # Feature flag
```

### Robot Configuration
```python
ROBOT = "UR10e"
GRIPPER = "Robotiq 2F-85 Gripper (Mechanism)"
VALID_OBJECTS = [
    'test_tube_blood', 'test_tube_DNA', 'test_tube_phenol',
    'beaker_water', 'dropper_Phenolphtalein',
    'bunsen_burner', 'biohazard_bin'
]
```

---

## Documentation

### Detailed Architecture Documentation
See `reports/ARCHITECTURE_ENHANCEMENT_PLAN.md` for:
- Complete agent specifications
- Data flow diagrams
- Implementation details
- Design patterns used
- Performance considerations
- Future enhancements

### API Documentation

#### Agent Input/Output Types

All agents use Pydantic models for type-safe data exchange:

```python
# Intent Router
class RouterInput(BaseModel):
    user_prompt: str
    valid_objects: List[str]

class RouterDecision(BaseModel):
    route: Literal["EXACT_MATCH", "AMBIGUOUS", "NOVEL_TASK", "NOT_TASK"]
    intent_text: Optional[str]
    distance: Optional[float]
    candidates: Optional[List[Tuple[str, float]]]

# Action Extractor
class ActionExtractionResult(BaseModel):
    actions: List[str]  # e.g., ["pick", "place", "pour"]
    reasoning: str

# Plan Generator
class PlanGenerationResult(BaseModel):
    plan_json: str
    success: bool
    error: Optional[str]
```

---

## Troubleshooting

### Common Issues

1. **"Ollama server not running"**
   - Solution: Start Ollama with `ollama serve` or launch the Ollama app

2. **"Model not found"**
   - Solution: Pull the model with `ollama pull llama3:8b`

3. **"RoboDK connection failed"**
   - Solution: Ensure RoboDK is open and the station is loaded

4. **"Object not found"**
   - Solution: Check that object names in your RoboDK station match `Config.VALID_OBJECTS`

5. **"Action extraction failing"**
   - Solution: Check that `procedural_api.json` exists and has valid action metadata

### Debug Mode

Enable verbose logging in `config/config.py`:

```python
DEBUG = True
LOG_LEVEL = "DEBUG"
```

---

## Contributing

### Adding New Actions

1. Add action metadata to `data/knowledge/procedural_api.json`:
```json
{
  "function_signature": "new_action(param: type)",
  "action_name": "new_action",
  "action_keywords": ["keyword1", "keyword2", "..."],
  "description": "What this action does",
  "constraints": {},
  "safety_critical": true/false,
  "example_usage": ["new_action('example')"]
}
```

2. Implement the action in `core/robot_control.py`

3. Update system prompt in `config/prompts.py` if needed

### Adding New Agents

1. Create agent file in `core/agents/`
2. Define Pydantic models in `core/agents/types.py`
3. Initialize agent in `DualRAGStrategy.__init__()`
4. Call agent in appropriate state handler
5. Add unit tests in `tests/`

---

## License

This project is developed for research purposes as part of a Master's thesis on multi-agent LLM architectures for robotic control.

---

## Citation

If you use this work in your research, please cite:

```
@mastersthesis{robot-agentic-rag-2026,
  title={Multi-Agent LLM Architecture for Robotic Arm Control},
  author={[Your Name]},
  year={2026},
  school={[Your University]}
}
```

---

## Acknowledgments

- **RoboDK** for the robot simulation platform
- **Ollama** for local LLM hosting
- **Anthropic** for Claude and LLM research
- **Meta** for Llama 3 model

---

## Contact

For questions, issues, or contributions, please open an issue on GitHub or contact [your email].