# ComfyUI LLM Toolkit

A custom node collection for integrating various LLM (Large Language Model) providers with ComfyUI.

## Features

- Text generation using various LLM providers (OpenAI and local models, etc.)
- Provider selection and configuration with dynamic model fetching
- API key management
- Seamless integration with ComfyUI workflows
- True Context-to-Context node connections with a unified wildcard interface

## Installation

1. Clone this repository into your ComfyUI custom_nodes directory:
   ```bash
   cd /path/to/ComfyUI/custom_nodes
   git clone https://github.com/yourusername/comfyui-llm-toolkit.git
   ```

2. Install the required dependencies:
   ```bash
   cd comfyui-llm-toolkit
   pip install -r requirements.txt
   ```

3. Restart ComfyUI

## Dependencies

The following Python packages are required:
- aiohttp
- pyyaml
- python-dotenv
- requests

## Configuration

1. API keys for various providers can be stored in a `.env` file in the root directory:
   ```
   OPENAI_API_KEY=your_openai_key
   # Add other provider keys as needed
   ```

2. Alternatively, you can provide API keys directly in the node interface.

## Usage

### LLM Provider Selector Node
Use this node to select the LLM provider and model. It outputs provider configuration within the wildcard "context" output.

- When you change the provider, the model dropdown updates dynamically with available models
- IP and Port fields appear/disappear based on whether the provider needs them
- The output is a single "context" type that contains the provider configuration

### Generate Text Node
Basic text generation with the selected provider and model.

- Connect it to the LLM Provider Selector by connecting their "context" ports
- The node automatically detects if provider config is present in the input
- You can override the model with the dropdown
- The "context" output contains both the original input data and the LLM response

## Unified ONE-INPUT / ONE-OUTPUT Architecture

The LLM Toolkit uses a single "context" input/output approach for maximum flexibility:

1. **Single Connection Point**: Each node has just one wildcard input and one wildcard output named "context"
2. **Smart Data Handling**: 
   - Provider config is embedded within the "context" data structure
   - Each node intelligently extracts the data it needs from the "context" input
   - Nodes preserve all input data and add their own data to the "context" output
3. **Cascading Data Flow**: As data flows through nodes, it accumulates in the "context" structure

For example, with nodes A → B → C:
- Node A creates an "context" with provider config
- Node B receives A's "context", extracts the provider config, and adds LLM response to the "context"
- Node C receives B's "context" which now contains both provider config and LLM response

This allows you to:
- Chain multiple LLM operations with a single connection
- Preserve and accumulate data throughout the workflow
- Easily integrate with other ComfyUI nodes

## Supported Providers

- OpenAI (Default: gpt-4o-mini)
- Ollama (local)
- And more... (Comming Soon)

## Troubleshooting

If you encounter model list update issues:
1. Make sure ComfyUI is running with the correct server configuration
2. Check that JavaScript is enabled in your browser
3. Verify that your API keys are correctly set in the .env file or provided in the node

If you encounter import errors:
1. Make sure you've installed all dependencies: `pip install -r requirements.txt`
2. Verify that you've placed the custom node in the correct directory
3. Restart ComfyUI after installation

## Testing Your Installation

Run the included test script to verify your setup:
```bash
cd /path/to/ComfyUI/custom_nodes/comfyui-llm-toolkit
python test_js_integration.py
```
