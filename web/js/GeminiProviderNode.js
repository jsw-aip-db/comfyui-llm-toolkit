import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

app.registerExtension({
    name: "ComfyUI.LLMToolkit.GeminiProviderNode",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "GeminiProviderNode") {
            return;
        }

        // Store original getExtraMenuOptions if it exists
        const originalGetExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;

        // Override the onNodeCreated to load models when node is created
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = async function() {
            const result = onNodeCreated?.apply(this, arguments);
            
            // Find the model widget
            const modelWidget = this.widgets?.find(w => w.name === "llm_model");
            if (!modelWidget) {
                console.warn("GeminiProviderNode: llm_model widget not found");
                return result;
            }

            // Store that models haven't been fetched yet
            this._modelsFetched = false;
            this._fetchingModels = false;

            // Function to fetch and update models
            const fetchAndUpdateModels = async () => {
                if (this._fetchingModels) return; // Prevent duplicate fetches
                this._fetchingModels = true;

                try {
                    console.log("GeminiProviderNode: Fetching models from server...");
                    const response = await fetch("/ComfyLLMToolkit/get_gemini_models");
                    
                    if (response.ok) {
                        const models = await response.json();
                        
                        if (Array.isArray(models) && models.length > 0) {
                            console.log(`GeminiProviderNode: Loaded ${models.length} models`);
                            
                            // Update the widget options
                            modelWidget.options.values = models;
                            
                            // If current value is not in the list, set to first model
                            if (!models.includes(modelWidget.value)) {
                                modelWidget.value = models[0];
                            }
                            
                            this._modelsFetched = true;
                        }
                    }
                } catch (error) {
                    console.error("GeminiProviderNode: Failed to fetch models:", error);
                } finally {
                    this._fetchingModels = false;
                }
            };

            // Only fetch models when the node is actually being used (connected or executed)
            // We'll fetch on first interaction with the widget
            const originalCallback = modelWidget.callback;
            modelWidget.callback = async function() {
                if (!this._modelsFetched && !this._fetchingModels) {
                    await fetchAndUpdateModels();
                }
                if (originalCallback) {
                    originalCallback.apply(this, arguments);
                }
            }.bind(this);

            // Also fetch when the node is selected (user clicks on it)
            const originalOnSelected = this.onSelected;
            this.onSelected = async function() {
                if (!this._modelsFetched && !this._fetchingModels) {
                    await fetchAndUpdateModels();
                }
                if (originalOnSelected) {
                    originalOnSelected.apply(this, arguments);
                }
            }.bind(this);

            return result;
        };

        // Add a "Refresh Models" option to the context menu
        nodeType.prototype.getExtraMenuOptions = function(canvas, node) {
            const options = originalGetExtraMenuOptions ? 
                originalGetExtraMenuOptions.apply(this, arguments) : [];

            options.push({
                content: "🔄 Refresh Gemini Models",
                callback: async () => {
                    const modelWidget = this.widgets?.find(w => w.name === "llm_model");
                    if (!modelWidget) return;

                    try {
                        console.log("GeminiProviderNode: Refreshing models...");
                        const response = await fetch("/ComfyLLMToolkit/get_gemini_models");
                        
                        if (response.ok) {
                            const models = await response.json();
                            
                            if (Array.isArray(models) && models.length > 0) {
                                console.log(`GeminiProviderNode: Refreshed ${models.length} models`);
                                modelWidget.options.values = models;
                                
                                if (!models.includes(modelWidget.value)) {
                                    modelWidget.value = models[0];
                                }
                                
                                this._modelsFetched = true;
                                app.canvas.setDirty(true);
                            }
                        }
                    } catch (error) {
                        console.error("GeminiProviderNode: Failed to refresh models:", error);
                    }
                }
            });

            return options;
        };
    }
});