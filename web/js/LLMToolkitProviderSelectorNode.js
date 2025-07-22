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
                const externalApiKeyWidget = this.widgets.find((w) => w.name === "external_api_key");

                const targetModelIndex = 1; // Desired index (0=provider, 1=model)

                // 1. Find and remove the original string widget
                const originalLlmModelWidget = this.widgets.find((w) => w.name === "llm_model");
                const originalIndex = this.widgets.findIndex((w) => w === originalLlmModelWidget);
                if (originalIndex !== -1) {
                    this.widgets.splice(originalIndex, 1);
                    console.log("LLM Toolkit Provider Node: Removed original llm_model string widget.");
                } else {
                    console.warn("LLM Toolkit Provider Node: Original llm_model widget not found for removal.");
                }

                // 2. Create the new COMBO widget using addWidget (this appends it)
                const llmModelCombo = this.addWidget(
                    "COMBO",
                    "llm_model",
                    "Provider not selected or models not fetched",
                    (value) => { this.properties['llm_model'] = value; },
                    { values: ["Provider not selected or models not fetched"] }
                );
                if (!this.properties) { this.properties = {}; }
                this.properties['llm_model'] = llmModelCombo.value; // Initialize property

                // 3. Move the newly added widget (currently last) to the target index
                const currentIndex = this.widgets.indexOf(llmModelCombo); // Find where addWidget put it
                if (currentIndex !== -1 && currentIndex !== targetModelIndex) {
                    // Remove from current position
                    this.widgets.splice(currentIndex, 1);
                    // Insert at target position (ensure index is valid)
                    const insertionIndex = Math.min(targetModelIndex, this.widgets.length);
                    this.widgets.splice(insertionIndex, 0, llmModelCombo);
                    console.log(`LLM Toolkit Provider Node: Moved llm_model combo widget to index ${insertionIndex}.`);
                } else if (currentIndex === -1) {
                    console.warn("LLM Toolkit Provider Node: Added llm_model combo widget not found for move.");
                }

                // Ensure property link is updated after potential reordering/widget recreation
                 const originalCallback = llmModelCombo.callback;
                 llmModelCombo.callback = (value) => {
                    if(originalCallback) originalCallback.call(this, value);
                    this.properties['llm_model'] = value;
                 }

                const updateLLMModels = async () => {
                    // Use the llmModelCombo reference (which should now be correctly placed)
                    if (!llmProviderWidget || !baseIpWidget || !portWidget || !llmModelCombo) {
                        console.warn("LLM Toolkit Provider Node: Required widgets not found (using new combo).");
                        return;
                    }

                    const currentModelValue = llmModelCombo.value;

                    llmModelCombo.options.values = ["Fetching models..."];
                    llmModelCombo.value = "Fetching models...";
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
                            llmModelCombo.options.values = models;
                            if (models.includes(currentModelValue)) {
                                llmModelCombo.value = currentModelValue;
                            } else {
                                llmModelCombo.value = models[0];
                            }
                        } else {
                            llmModelCombo.options.values = ["No models found or Error"];
                            llmModelCombo.value = "No models found or Error";
                            console.warn("No valid models received or error fetching.");
                        }
                        this.properties['llm_model'] = llmModelCombo.value;

                    } catch (error) {
                        console.error("Error updating LLM models:", error);
                        llmModelCombo.options.values = ["Error fetching models"];
                        llmModelCombo.value = "Error fetching models";
                        this.properties['llm_model'] = llmModelCombo.value;
                    } finally {
                        this.setDirtyCanvas(true, true);
                    }
                };

                [llmProviderWidget, baseIpWidget, portWidget, externalApiKeyWidget].forEach(widget => {
                    if (widget) {
                        const originalCallback = widget.callback;
                        widget.callback = async (value) => {
                            if (originalCallback) {
                                originalCallback.call(this, value);
                            }
                            if (widget.name && this.properties && widget.name in this.properties) {
                                this.properties[widget.name] = value;
                            }
                            await updateLLMModels();
                        };
                    }
                });

                setTimeout(updateLLMModels, 100);
            };
        }
    }
}); 