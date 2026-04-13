# global configuration
class Config:
    # Robot Configuration
    ROBOT = "UR10e"
    GRIPPER = "Robotiq 2F-85 Gripper (Mechanism)"
    TOOL = "Tool 1"
    COLLISION_ACTIVE = False
    
    # Movement Parameters
    GRIP_Z = 0   # Grab exactly at the center height of the cube
    LIFT_Z = 200 # How high to lift after grabbing
    TABLE_Z = 0   # Absolute Z of table surface
    SHIFT_STEP = 100 # distance to shift when finding a new spot
    
    # Storage Area (Rectangular Region)
    # The robot will search for a free spot within these bounds
    STORAGE_AREA = {
        "min_x": 0,
        "max_x": 300,
        "min_y": 200,
        "max_y": 300,
    }

    # LLM Configuration - Multi-Model Architecture
    LLM_MODEL = "mistral:7b-instruct"  # For semantic reasoning & routing (Intent Router, Action Extractor, etc.)
    LLM_PLAN_GENERATION_MODEL = "llama3:8b"  # For structured JSON generation (Plan Generator)
    LLM_TEMPERATURE = 0.0  # Deterministic for reproducibility in experiments
    LLM_TIMEOUT = 120  # Seconds

    # Valid Scene Objects
    VALID_OBJECTS = [
        'test_tube_blood', 'test_tube_DNA', 'test_tube_phenol',
        'test_tube_hydrochloric_acid', 'test_tube_sodium_hydroxide',
        'test_tube_empty', 'beaker_water', 'dropper_Phenolphtalein',
        'bunsen_burner', 'biohazard_bin'
    ]

    # Multi-Agent Configuration (Balanced thresholds for multi-stage matching)
    INTENT_ROUTER_THRESHOLD_EXACT = 1.0      # Distance < 1.0 = Exact match (was 1.1)
    INTENT_ROUTER_THRESHOLD_AMBIGUOUS = 1.5  # 1.0 <= Distance < 1.5 = Ambiguous (was 1.6)

    # Procedural Retrieval Configuration
    PROCEDURAL_MIN_RESULTS = 3   # Minimum APIs to retrieve

    # Action Extraction Configuration
    ACTION_EXTRACTION_ENABLED = True  # Feature flag for action extraction
