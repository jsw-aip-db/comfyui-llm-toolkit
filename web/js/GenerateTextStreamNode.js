// GenerateTextNodeStream.js
import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js"; // Import api for event listener
import { ComfyWidgets } from "/scripts/widgets.js";

app.registerExtension({
    // Unique name for this extension
    name: "Comfy.GenerateTextNodeStream",

    async setup(app) {
        // --- Add WebSocket listeners ---
        api.addEventListener("llmtoolkit.stream.start", ({ detail }) => {
            // Find the node using the ID from the message
            const node = app.graph.getNodeById(detail.node);
            if (node && node.__llmResponseWidget) {
                // Clear the widget when a new stream starts for this node
                node.__llmResponseWidget.value = "";
                if (node.__llmResponseWidget.inputEl) node.__llmResponseWidget.inputEl.value = "";
                 app.graph.setDirtyCanvas(true);
            }
        });

        api.addEventListener("llmtoolkit.stream.chunk", ({ detail }) => {
            // Find the node using the ID from the message
            const node = app.graph.getNodeById(detail.node);
            if (node && node.__llmResponseWidget && detail.text) {
                // Append the received text chunk
                node.__llmResponseWidget.value += detail.text;
                if (node.__llmResponseWidget.inputEl) node.__llmResponseWidget.inputEl.value = node.__llmResponseWidget.value; // Ensure inputEl also updates
                 app.graph.setDirtyCanvas(true); // Redraw needed? Maybe not for every chunk.
            }
        });

         api.addEventListener("llmtoolkit.stream.end", ({ detail }) => {
            const node = app.graph.getNodeById(detail.node);
            if (node && node.__llmResponseWidget && detail.final_text !== undefined) {
                 // Optional: Ensure final text matches exactly
                 // node.__llmResponseWidget.value = detail.final_text;
                 // if (node.__llmResponseWidget.inputEl) node.__llmResponseWidget.inputEl.value = detail.final_text;
                 console.log(`Stream ended for node ${detail.node}`);
                 app.graph.setDirtyCanvas(true);
            }
        });

        api.addEventListener("llmtoolkit.stream.error", ({ detail }) => {
            const node = app.graph.getNodeById(detail.node);
            if (node && node.__llmResponseWidget && detail.error) {
                 // Append or display the error prominently
                 const currentVal = node.__llmResponseWidget.value;
                 const errorMsg = `\n\n--- STREAM ERROR ---\n${detail.error}`;
                 node.__llmResponseWidget.value = currentVal + errorMsg;
                if (node.__llmResponseWidget.inputEl) node.__llmResponseWidget.inputEl.value = node.__llmResponseWidget.value;
                console.error(`Stream error for node ${detail.node}: ${detail.error}`);
                 app.graph.setDirtyCanvas(true);
            }
        });
    },

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Only act on our new streaming node
        if (nodeData.name !== "LLMToolkitTextGeneratorStream") return; // <-- Target new node name

        /* -----------------------------
           1. Inject widget on creation (Same as before)
        ----------------------------- */
        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const ret = origOnNodeCreated ? origOnNodeCreated.apply(this, arguments) : undefined;

            // Check if widget already exists (e.g., on graph load)
            if (!this.widgets?.find(w => w.name === `${nodeData.name}_response`)) {
                // Use a consistent widget name per node *type* if needed,
                // or ensure it's recreated correctly on load if instance-specific.
                // Let's use a consistent name for simplicity here.
                const widgetName = `${nodeData.name}_response`;

                const widgetInfo = ComfyWidgets.STRING(
                    this,
                    widgetName,
                    ["STRING", {
                        default: "",
                        placeholder: "LLM response will stream here...", // Updated placeholder
                        multiline: true,
                    }],
                    app
                );

                widgetInfo.widget.inputEl.readOnly = true;
                this.__llmResponseWidget = widgetInfo.widget || widgetInfo; // Store reference

                // Ensure node size reflects the added widget if it wasn't there before
                // This might require careful handling during graph load vs new node creation
                 this.setSize(this.computeSize());
            } else {
                // Widget exists, likely from graph load, find and store reference
                 this.__llmResponseWidget = this.widgets.find(w => w.name === `${nodeData.name}_response`);
            }

            return ret;
        };

        /* -----------------------------
           2. Hook into onExecuted (Optional - for final state/saving)
        ----------------------------- */
        // onExecuted still receives the *final* data when the python function returns.
        // We can use it to ensure the widget's value is set correctly for saving the workflow.
        const origOnExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            if (origOnExecuted) origOnExecuted.apply(this, arguments);

            let final_text = "";
            if (message?.ui?.string) {
                final_text = message.ui.string.join("\n");
            } else if (message?.string) {
                 final_text = message.string.join("\n");
            } else if (Array.isArray(message)) {
                final_text = message.join("\n");
            } else if (typeof message === 'string') {
                 final_text = message;
            }


            if (this.__llmResponseWidget) {
                 // Set the final value, ensuring it's accurate for saving state
                 const finalValue = final_text.trim();
                 this.__llmResponseWidget.value = finalValue;
                 if (this.__llmResponseWidget.inputEl) this.__llmResponseWidget.inputEl.value = finalValue;
                  app.graph.setDirtyCanvas(true);
            }
        };

        /* -----------------------------
           3. Restore saved value on load (Same as before)
        ----------------------------- */
        const origOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (config) {
            // Ensure the widget exists before configuring
            if (!this.__llmResponseWidget) {
                 // Try to find it again if onNodeCreated didn't catch it during load
                 this.__llmResponseWidget = this.widgets?.find(w => w.name === `${nodeData.name}_response`);
            }

            if (origOnConfigure) origOnConfigure.apply(this, arguments);

            // Now configure if the widget reference is valid
            if (this.__llmResponseWidget && config?.widgets_values?.length) {
                const widgetName = `${nodeData.name}_response`;
                const idx = this.widgets?.findIndex(w => w.name === widgetName); // Find by name
                if (idx !== -1 && config.widgets_values.length > idx) {
                    const savedVal = config.widgets_values[idx] || "";
                    // Restore value from saved workflow state
                    this.__llmResponseWidget.value = savedVal;
                    if (this.__llmResponseWidget.inputEl) this.__llmResponseWidget.inputEl.value = savedVal;
                }
            }
        };

         /* -----------------------------
           4. Ensure widget reference on graph load (Added robustness)
           ----------------------------- */
         const origOnAdded = nodeType.prototype.onAdded;
         nodeType.prototype.onAdded = function() {
              if(origOnAdded) origOnAdded.apply(this, arguments);
              // Ensure the widget reference is set after the node is added to the graph
              // This helps catch cases where onNodeCreated might race with graph loading.
               if (!this.__llmResponseWidget) {
                   this.__llmResponseWidget = this.widgets?.find(w => w.name === `${nodeData.name}_response`);
               }
         }
    },
});