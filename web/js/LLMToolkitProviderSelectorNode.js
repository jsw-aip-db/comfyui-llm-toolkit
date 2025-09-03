// Minimal safe version to prevent freezes
import { app } from "/scripts/app.js";

app.registerExtension({
    name: "Comfy.LLMToolkitProviderSelector",

    async beforeRegisterNodeDef(nodeType, nodeData, appInstance) {
        if (nodeData.name === "LLMToolkitProviderSelector") {
            console.log("LLMToolkitProviderSelector: Using minimal safe mode");
            
            const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
            
            nodeType.prototype.onNodeCreated = function() {
                // Call original if exists
                if (originalOnNodeCreated) {
                    try {
                        originalOnNodeCreated.apply(this, arguments);
                    } catch (e) {
                        console.error("LLMToolkitProviderSelector: Error in original onNodeCreated:", e);
                    }
                }
                
                // Just log that we're here, don't modify widgets or make fetch calls
                console.log("LLMToolkitProviderSelector: Node created safely without modifications");
                
                // Ensure widgets exist but don't modify them
                if (!this.widgets) {
                    this.widgets = [];
                }
            };
        }
    }
});