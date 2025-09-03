# test_api_key_context.py
import os
import sys
import logging
from typing import Any, Dict, Optional, Tuple

# Ensure parent directory is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from context_payload import extract_context

logger = logging.getLogger(__name__)

class TestAPIKeyContext:
    """
    A test node to verify that API key context handling doesn't interfere with other providers.
    This node displays what API keys are available in the context and what the active provider config is.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "context": ("*", {"tooltip": "Context to inspect for API keys"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "test_context"
    CATEGORY = "🔗llm_toolkit/debug"
    OUTPUT_NODE = True

    def test_context(self, context: Optional[Dict[str, Any]] = None) -> Tuple[str]:
        """
        Tests and reports on API key context handling.
        
        Args:
            context: Context to inspect
        
        Returns:
            A report string showing API key status
        """
        logger.info("TestAPIKeyContext: Analyzing context...")
        
        # Initialize or copy context
        if context is None:
            output_context = {}
        elif isinstance(context, dict):
            output_context = context.copy()
        else:
            # Handle ContextPayload or other formats
            unwrapped = extract_context(context)
            if isinstance(unwrapped, dict):
                output_context = unwrapped.copy()
            else:
                output_context = {}

        report_lines = ["=== API Key Context Test Report ===\n"]
        
        # Check for context-based API keys
        context_api_keys = output_context.get("api_keys", {})
        if context_api_keys:
            report_lines.append("📋 Context API Keys Found:")
            for provider, key in context_api_keys.items():
                masked_key = key[:5] + "..." if len(key) > 5 else "..."
                report_lines.append(f"  • {provider}: {masked_key}")
        else:
            report_lines.append("📋 No context API keys found")
        
        report_lines.append("")
        
        # Check provider config
        provider_config = output_context.get("provider_config", {})
        if provider_config:
            provider_name = provider_config.get("provider_name", "unknown")
            config_key = provider_config.get("api_key", "")
            masked_config_key = config_key[:5] + "..." if len(config_key) > 5 else "none"
            
            report_lines.append("⚙️ Provider Config:")
            report_lines.append(f"  • Provider: {provider_name}")
            report_lines.append(f"  • API Key: {masked_config_key}")
            
            # Check if context key would override
            if context_api_keys:
                context_key = context_api_keys.get(provider_name)
                if not context_key and provider_name == "google":
                    context_key = context_api_keys.get("gemini")
                elif not context_key and provider_name == "gemini":
                    context_key = context_api_keys.get("google")
                
                if context_key:
                    masked_context_key = context_key[:5] + "..." if len(context_key) > 5 else "..."
                    report_lines.append(f"  • Context Override: {masked_context_key} ✅")
                else:
                    report_lines.append(f"  • Context Override: none")
        else:
            report_lines.append("⚙️ No provider config found")
        
        report_lines.append("")
        
        # Check for potential conflicts
        report_lines.append("🔍 Compatibility Check:")
        
        if context_api_keys and provider_config:
            provider_name = provider_config.get("provider_name", "").lower()
            matching_keys = []
            
            for ctx_provider in context_api_keys.keys():
                if ctx_provider.lower() == provider_name or \
                   (ctx_provider == "google" and provider_name == "gemini") or \
                   (ctx_provider == "gemini" and provider_name == "google"):
                    matching_keys.append(ctx_provider)
            
            if matching_keys:
                report_lines.append(f"  • Found matching context key(s): {', '.join(matching_keys)} ✅")
            else:
                report_lines.append(f"  • No matching context keys for provider '{provider_name}' ✅")
                report_lines.append("  • This won't interfere with the provider's API key")
        else:
            report_lines.append("  • No potential conflicts detected ✅")
        
        report = "\n".join(report_lines)
        logger.info("TestAPIKeyContext: Report generated")
        print(report)  # Also print to console for debugging
        
        return (report,)


# --- Node Mappings ---
NODE_CLASS_MAPPINGS = {
    "TestAPIKeyContext": TestAPIKeyContext
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "TestAPIKeyContext": "Test API Key Context (LLMToolkit Debug)"
}