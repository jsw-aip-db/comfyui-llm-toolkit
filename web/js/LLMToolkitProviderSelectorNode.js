// LLMToolkitProviderSelectorNode.js
import { app } from "/scripts/app.js";

app.registerExtension({
    name: "Comfy.LLMToolkitProviderSelector",

    async beforeRegisterNodeDef(nodeType, nodeData, appInstance) {
        if (nodeData.name === "LLMToolkitProviderSelector") {
            const originalOnNodeCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function() {
                if (originalOnNodeCreated) {
                    originalOnNodeCreated.apply(this, arguments);
                }

                const llmProviderWidget = this.widgets.find((w) => w.name === "llm_provider");
                const baseIpWidget = this.widgets.find((w) => w.name === "base_ip");
                const portWidget = this.widgets.find((w) => w.name === "port");
                const llmModelWidget = this.widgets.find((w) => w.name === "llm_model");
                const externalApiKeyWidget = this.widgets.find((w) => w.name === "external_api_key");

                const updateLLMModels = async () => {
                    if (!llmProviderWidget || !baseIpWidget || !portWidget || !llmModelWidget) {
                        console.warn("LLM Toolkit Provider Node: Required widgets not found.");
                        return;
                    }

                    const currentModelValue = llmModelWidget.value; // Store current value

                    // Show fetching status
                    llmModelWidget.options.values = ["Fetching models..."];
                    llmModelWidget.value = "Fetching models...";
                    this.setDirtyCanvas(true, true);

                    try {
                        console.log(`Fetching models for ${llmProviderWidget.value}...`);
                        const response = await fetch("/ComfyLLMToolkit/get_provider_models", {
                            method: "POST",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify({
                                llm_provider: llmProviderWidget.value,
                                base_ip: baseIpWidget.value,
                                port: portWidget.value,
                                external_api_key: externalApiKeyWidget?.value || ""
                            })
                        });

                        if (!response.ok) {
                            throw new Error(`HTTP error! status: ${response.status}`);
                        }

                        const models = await response.json();
                        console.log("Fetched models:", models);

                        if (Array.isArray(models) && models.length > 0 && models[0] !== "Error fetching models" && models[0] !== "No models found") {
                            llmModelWidget.options.values = models;
                            // Try to restore previous value if it exists in the new list, otherwise set to the first model
                            if (models.includes(currentModelValue)) {
                                llmModelWidget.value = currentModelValue;
                            } else {
                                llmModelWidget.value = models[0];
                            }
                        } else {
                            llmModelWidget.options.values = ["No models found or Error"];
                            llmModelWidget.value = "No models found or Error";
                            console.warn("No valid models received or error fetching.");
                        }
                    } catch (error) {
                        console.error("Error updating LLM models:", error);
                        llmModelWidget.options.values = ["Error fetching models"];
                        llmModelWidget.value = "Error fetching models";
                    } finally {
                        this.setDirtyCanvas(true, true); // Ensure UI update
                    }
                };

                // Add callbacks to update models when provider, IP, port, or key changes
                [llmProviderWidget, baseIpWidget, portWidget, externalApiKeyWidget].forEach(widget => {
                    if (widget) {
                        const originalCallback = widget.callback;
                        widget.callback = async (value) => {
                            if (originalCallback) {
                                originalCallback.call(this, value); // Call original callback if exists
                            }
                            await updateLLMModels(); // Then update models
                        };
                    }
                });

                // Initial model fetch when the node is created or loaded
                setTimeout(updateLLMModels, 100); // Delay slightly to ensure widgets are ready
            };
        }
    }
}); 