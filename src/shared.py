import threading
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class LogEntry:
    timestamp: float
    message: str
    color: str

class SharedState:
    def __init__(self):
        self.current_task: str = "Initializing..."
        self.current_subtask: str = "None"
        self.logs: List[LogEntry] = []
        self.reasoning: str = ""
        self.model_output: str = ""
        self.code_generated: str = ""
        self.start_time: float = 0.0
        
        # Flags for flow control
        self.interrupt_event = threading.Event()  # Set by Web when user clicks "Interrupt"
        self.feedback_event = threading.Event()   # Set by Web when user submits feedback
        self.approval_event = threading.Event()   # Set by Web when waiting for approval
        self.feedback_text: str = ""              # Stores the feedback string
        self.is_waiting_for_feedback: bool = False
        self.is_waiting_for_approval: bool = False
        self.is_running: bool = False
        
    def add_log(self, message: str, color: str = "white"):
        import time
        self.logs.append(LogEntry(time.time(), str(message), color))
        # Keep log size manageable
        if len(self.logs) > 500:
            self.logs.pop(0)

# Global singleton instance
state_manager = SharedState()
