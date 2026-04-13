"""
Ollama Token Tracker for RQ3

Wraps ollama.chat calls to track actual token usage from Llama/Mistral models.
"""

import ollama
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class OllamaCallStats:
    """Statistics for a single Ollama call"""
    model: str
    prompt_tokens: int  # prompt_eval_count
    completion_tokens: int  # eval_count
    total_tokens: int
    response_time_ms: float  # total_duration in ms


@dataclass
class OllamaSessionTracker:
    # tracks all ollama calls in a session
    calls: List[OllamaCallStats] = field(default_factory=list)

    def add_call(self, stats: OllamaCallStats):
        # add a call to the tracker
        self.calls.append(stats)

    def get_total_tokens(self) -> int:
        # get total tokens across all calls
        return sum(call.total_tokens for call in self.calls)

    def get_total_prompt_tokens(self) -> int:
        # get total prompt tokens across all calls
        return sum(call.prompt_tokens for call in self.calls)

    def get_total_completion_tokens(self) -> int:
        # get total completion tokens across all calls
        return sum(call.completion_tokens for call in self.calls)

    def get_total_time_ms(self) -> float:
        # get total response time across all calls
        return sum(call.response_time_ms for call in self.calls)

    def get_breakdown_by_model(self) -> Dict[str, Dict[str, int]]:
        # get token breakdown by model
        breakdown = {}
        for call in self.calls:
            if call.model not in breakdown:
                breakdown[call.model] = {
                    'prompt_tokens': 0,
                    'completion_tokens': 0,
                    'total_tokens': 0,
                    'calls': 0
                }
            breakdown[call.model]['prompt_tokens'] += call.prompt_tokens
            breakdown[call.model]['completion_tokens'] += call.completion_tokens
            breakdown[call.model]['total_tokens'] += call.total_tokens
            breakdown[call.model]['calls'] += 1
        return breakdown

    def reset(self):
        # reset all tracked calls
        self.calls.clear()

    def __str__(self):
        # string representation
        if not self.calls:
            return "No calls tracked"

        breakdown = self.get_breakdown_by_model()
        lines = [f"Total Calls: {len(self.calls)}"]
        for model, stats in breakdown.items():
            lines.append(f"  {model}: {stats['total_tokens']} tokens ({stats['calls']} calls)")
        return "\n".join(lines)


# Global tracker instance
_global_tracker: Optional[OllamaSessionTracker] = None


def get_tracker() -> OllamaSessionTracker:
    # get or create the global tracker
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = OllamaSessionTracker()
    return _global_tracker


def reset_tracker():
    # reset the global tracker
    global _global_tracker
    _global_tracker = OllamaSessionTracker()


def tracked_chat(model: str, messages: List[Dict[str, str]], **kwargs) -> Any:
    """
    Wrapper around ollama.chat that tracks token usage.

    Args:
        model: Model name
        messages: Chat messages
        **kwargs: Additional arguments to ollama.chat

    Returns:
        ollama.ChatResponse
    """
    # Call original ollama.chat (not the patched version)
    original_chat = getattr(ollama, '_original_chat', None)
    if original_chat is None:
        # If not patched, use the actual ollama.chat from the module
        import ollama as ollama_module
        original_chat = ollama_module.Client().chat

    response = original_chat(model=model, messages=messages, **kwargs)

    # Extract token counts from response
    prompt_tokens = getattr(response, 'prompt_eval_count', 0) or 0
    completion_tokens = getattr(response, 'eval_count', 0) or 0
    total_tokens = prompt_tokens + completion_tokens

    # Extract timing (convert nanoseconds to milliseconds)
    total_duration_ns = getattr(response, 'total_duration', 0) or 0
    total_duration_ms = total_duration_ns / 1_000_000

    # Create stats
    stats = OllamaCallStats(
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        response_time_ms=total_duration_ms
    )

    # Add to tracker
    tracker = get_tracker()
    tracker.add_call(stats)

    return response


def patch_ollama():
    """
    Patch ollama.chat to use tracked_chat.
    Call this at the start of your test to enable tracking.
    """
    import ollama
    ollama._original_chat = ollama.chat
    ollama.chat = tracked_chat


def unpatch_ollama():
    """
    Restore original ollama.chat.
    Call this at the end of your test to restore normal behavior.
    """
    import ollama
    if hasattr(ollama, '_original_chat'):
        ollama.chat = ollama._original_chat
        delattr(ollama, '_original_chat')


if __name__ == "__main__":
    # Test the tracker
    print("Testing Ollama Tracker...")

    # Patch ollama
    patch_ollama()
    reset_tracker()

    # Make a test call
    print("\nMaking test call to llama3:8b...")
    response = ollama.chat(
        model='llama3:8b',
        messages=[{'role': 'user', 'content': 'Say hello in 5 words'}]
    )
    print(f"Response: {response.message.content}")

    # Check tracker
    tracker = get_tracker()
    print(f"\nTracker stats:")
    print(f"  Total tokens: {tracker.get_total_tokens()}")
    print(f"  Prompt tokens: {tracker.get_total_prompt_tokens()}")
    print(f"  Completion tokens: {tracker.get_total_completion_tokens()}")
    print(f"  Time: {tracker.get_total_time_ms():.2f}ms")
    print(f"\n{tracker}")

    # Unpatch
    unpatch_ollama()
    print("\n✓ Test complete")