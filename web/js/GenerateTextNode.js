import { app } from "/scripts/app.js";
import { ComfyWidgets } from "/scripts/widgets.js";

/* ------------------------------------------------------------------
   Extension: Comfy.GenerateTextNode
   Adds a read‑only multiline STRING widget to the "Generate Text (LLMToolkit)"
   node that displays the LLM response directly on the node.  The node still
   outputs the normal "context" so it can be connected to an external
   Display Text node as before.
-------------------------------------------------------------------*/

app.registerExtension({
    name: "Comfy.GenerateTextNode",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Only act on our LLMToolkitTextGenerator node
        if (nodeData.name !== "LLMToolkitTextGenerator") return;

        /* -----------------------------
           1.  Inject widget on creation
        ----------------------------- */
        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            // Invoke original handler first
            const ret = origOnNodeCreated ? origOnNodeCreated.apply(this, arguments) : undefined;

            // Build a unique widget name so each node instance has its own key
            const existingNodes = app.graph._nodes.filter(n => n.type === nodeData.name);
            const widgetName = `${nodeData.name}_response_${existingNodes.length}`;

            // Create read‑only STRING widget for displaying response
            const widgetInfo = ComfyWidgets.STRING(
                this,
                widgetName,
                ["STRING", {
                    default: "",
                    placeholder: "LLM response will appear here …",
                    multiline: true,
                }],
                app
            );

            // Make read‑only
            widgetInfo.widget.inputEl.readOnly = true;

            // Store reference for quick updates
            this.__llmResponseWidget = widgetInfo.widget || widgetInfo;

            // Ensure node size grows to fit another widget
            this.setSize(this.computeSize());

            return ret;
        };

        /* -----------------------------
           2.  Helper to update widget
        ----------------------------- */
        function updateResponseWidget(textContainer) {
            if (!this.__llmResponseWidget) return;

            // Extract final text string (textContainer may be array/string)
            let valueToSet = "";
            if (Array.isArray(textContainer)) {
                valueToSet = textContainer.join("\n");
            } else if (typeof textContainer === "string") {
                valueToSet = textContainer;
            }

            // Trim and write
            valueToSet = valueToSet.trim();
            this.__llmResponseWidget.value = valueToSet;
            if (this.__llmResponseWidget.inputEl) this.__llmResponseWidget.inputEl.value = valueToSet;
            app.graph.setDirtyCanvas(true);
        }

        /* -----------------------------
           3.  Hook into onExecuted to receive message
        ----------------------------- */
        const origOnExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            if (origOnExecuted) origOnExecuted.apply(this, arguments);

            // message is typically { ui: { string: [...] }, result: (…) }
            let textData = undefined;
            if (message?.ui?.string) {
                textData = message.ui.string;
            } else if (message?.string) {
                textData = message.string;
            } else if (Array.isArray(message)) {
                textData = message;
            }
            updateResponseWidget.call(this, textData);
        };

        /* -----------------------------
           4. Restore saved value on load
        ----------------------------- */
        const origOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (config) {
            if (origOnConfigure) origOnConfigure.apply(this, arguments);

            if (!this.__llmResponseWidget || !config?.widgets_values?.length) return;

            const idx = this.widgets.findIndex(w => w === this.__llmResponseWidget);
            if (idx !== -1 && config.widgets_values.length > idx) {
                const savedVal = config.widgets_values[idx] || "";
                this.__llmResponseWidget.value = savedVal;
                if (this.__llmResponseWidget.inputEl) this.__llmResponseWidget.inputEl.value = savedVal;
            }
        };
    },
}); 