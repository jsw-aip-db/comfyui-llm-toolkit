import { app } from "/scripts/app.js";
import { ComfyWidgets } from "/scripts/widgets.js";

app.registerExtension({
    name: "Comfy.DisplayTextNode",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "Display_Text") {
            // Add output labels and set shapes
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const ret = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                // Set output labels
                if (this.outputs) {
                    // Ensure the number of outputs matches the expected labels
                    const expectedOutputCount = 5;
                    if (this.outputs.length >= expectedOutputCount) {
                        // Update labels to match Python RETURN_NAMES
                        this.outputs[0].name = "context";     // Corrected label for the first output
                        this.outputs[1].name = "text_list";
                        this.outputs[2].name = "count";
                        this.outputs[3].name = "selected";
                        this.outputs[4].name = "text_full";   // Added label for the fifth output

                        // Set the text_list output (index 1) to use grid shape
                        this.outputs[1].shape = LiteGraph.GRID_SHAPE;
                    } else {
                        console.error(`DisplayTextNode: Expected ${expectedOutputCount} outputs, but found ${this.outputs.length}. Cannot set labels.`);
                    }
                }

                let Display_Text = app.graph._nodes.filter(wi => wi.type == nodeData.name),
                    nodeName = `${nodeData.name}_${Display_Text.length}`;

                console.log(`Create ${nodeData.name}: ${nodeName}`);

                const wi = ComfyWidgets.STRING(
                    this,
                    nodeName,
                    ["STRING", {
                        default: "",
                        placeholder: "Message will appear here ...",
                        multiline: true,
                    }],
                    app
                );
                wi.widget.inputEl.readOnly = true;

                // Store a reference to the created widget on the node instance for easy access later
                this.__displayTextWidget = wi.widget || wi; // In case ComfyWidgets.STRING returns object containing widget

                return ret;
            };

            // Add tooltips and visual indicators
            const onDrawForeground = nodeType.prototype.onDrawForeground;
            nodeType.prototype.onDrawForeground = function (ctx) {
                const ret = onDrawForeground?.apply(this, arguments);
                
                if (this.outputs && this.outputs.length > 0) {
                    const outputLabels = [
                        "Complete Text",
                        "List of Lines",
                        "Line Count",
                        "Selected Line"
                    ];
                    const outputTooltips = [
                        "Full text content",
                        "Individual lines as separate outputs",
                        "Total number of non-empty lines",
                        "Currently selected line based on select input"
                    ];
                    
                    for (let i = 0; i < this.outputs.length; i++) {
                        const output = this.outputs[i];
                        output.tooltip = outputTooltips[i];
                    }
                }
                return ret;
            };

            // Function set value
            const outSet = function (texts) {
                if (!this.__displayTextWidget) {
                    console.error("DisplayTextNode: display text widget reference not found on node.");
                    return;
                }

                if (texts && texts.length > 0) {
                    let value_to_set = texts;

                    if (Array.isArray(value_to_set)) {
                        value_to_set = value_to_set
                            .filter(word => word != null)
                            .map(word => String(word).trim())
                            .join("\n");
                    } else {
                        value_to_set = String(value_to_set).trim();
                    }

                    // Update both widget value and the DOM element value if present
                    this.__displayTextWidget.value = value_to_set;
                    if (this.__displayTextWidget.inputEl) {
                        this.__displayTextWidget.inputEl.value = value_to_set;
                    }

                    app.graph.setDirtyCanvas(true);
                }
            };

            // onExecuted
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                // Log the entire message for debugging
                console.log("DisplayTextNode onExecuted message:", message);
                if (onExecuted) {
                    onExecuted.apply(this, arguments);
                }

                // Prefer the new structure { ui: { string: [...] } }
                let texts = undefined;
                if (message?.ui?.string) {
                    texts = message.ui.string;
                } else if (message?.string) {
                    // Fallback to flat structure (some nodes just send {string: [...]})
                    texts = message.string;
                } else if (Array.isArray(message)) {
                    // Some nodes may directly send the array
                    texts = message;
                }

                outSet.call(this, texts);
            };

            // onConfigure
            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function (w) {
                onConfigure?.apply(this, arguments);
                // Use stored reference if available
                if (this.__displayTextWidget && w?.widgets_values?.length) {
                    const widget_index = this.widgets.findIndex(widget => widget === this.__displayTextWidget);
                    if (widget_index !== -1 && w.widgets_values.length > widget_index) {
                        const saved_value = w.widgets_values[widget_index] || "";
                        this.__displayTextWidget.value = saved_value;
                        if (this.__displayTextWidget.inputEl) {
                            this.__displayTextWidget.inputEl.value = saved_value;
                        }
                        app.graph.setDirtyCanvas(true);
                    }
                }
            };
        }
    },
});