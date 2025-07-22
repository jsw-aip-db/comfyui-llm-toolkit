import { app } from "/scripts/app.js";

app.registerExtension({
    name: "Comfy.GroqProviderNode",

    async beforeRegisterNodeDef(nodeType, nodeData, appInstance) {
        if (nodeData.name !== "GroqProviderNode") return;

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function () {
            if (originalOnNodeCreated) originalOnNodeCreated.apply(this, arguments);

            // Locate original llm_model text widget and remove it
            const original = this.widgets.find((w) => w.name === "llm_model");
            const originalIndex = this.widgets.findIndex((w) => w === original);
            if (originalIndex !== -1) this.widgets.splice(originalIndex, 1);

            // Create combo widget and insert at index 0 (top)
            const combo = this.addWidget(
                "COMBO",
                "llm_model",
                "Select Groq model…",
                (v) => {
                    this.properties.llm_model = v;
                },
                { values: ["Fetching models…"] }
            );

            // Move to top if not already
            const idx = this.widgets.indexOf(combo);
            if (idx > 0) {
                this.widgets.splice(idx, 1);
                this.widgets.splice(0, 0, combo);
            }

            // Fetch Groq models once
            const fetchModels = async () => {
                try {
                    const res = await fetch("/ComfyLLMToolkit/get_groq_models");
                    if (!res.ok) throw new Error(res.statusText);
                    const models = await res.json();
                    if (Array.isArray(models) && models.length) {
                        combo.options.values = models;
                        // Default to first model in list
                        combo.value = models[0]; 
                        this.properties.llm_model = combo.value;
                    } else {
                        combo.options.values = ["No models found"];
                        combo.value = "No models found";
                    }
                } catch (e) {
                    console.error("GroqProviderNode: model fetch failed", e);
                    combo.options.values = ["Error fetching models"];
                    combo.value = "Error fetching models";
                } finally {
                    this.setDirtyCanvas(true, true);
                }
            };

            setTimeout(fetchModels, 100);
        };
    },
}); 