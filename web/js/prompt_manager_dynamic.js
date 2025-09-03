// Safe dynamic input ports for LLMPromptManager node
import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "LLMToolkit.PromptManagerDynamic",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "LLMPromptManager") {
            const origOnNodeCreated = nodeType.prototype.onNodeCreated;
            
            nodeType.prototype.onNodeCreated = function() {
                const ret = origOnNodeCreated ? origOnNodeCreated.apply(this, arguments) : undefined;
                
                // Initialize with just one context input
                if (!this.inputs || this.inputs.length === 0) {
                    this.inputs = [];
                    this.addInput("context", "*");
                }
                
                // Track state with safety flags
                this.isProcessing = false;
                this.maxInputs = 10; // Limit maximum inputs to prevent UI issues
                
                return ret;
            };
            
            // Override onConnectionsChange to handle dynamic inputs safely
            const origOnConnectionsChange = nodeType.prototype.onConnectionsChange;
            nodeType.prototype.onConnectionsChange = function(type, index, connected, link_info) {
                if (origOnConnectionsChange) {
                    origOnConnectionsChange.apply(this, arguments);
                }
                
                // Prevent processing during load or if already processing
                if (this.isProcessing || app.graph.isLoading) {
                    return;
                }
                
                if (type === 1) { // Input connection
                    this.isProcessing = true;
                    
                    try {
                        if (connected) {
                            // Check if this is the last available input and we're under the limit
                            const lastInputIndex = this.inputs.length - 1;
                            if (index === lastInputIndex && this.inputs.length < this.maxInputs) {
                                // Add a new input port
                                const nextIndex = this.inputs.length;
                                this.addInput(`context_${nextIndex}`, "*");
                                
                                // Update size to accommodate new input
                                const newSize = this.computeSize();
                                if (this.size) {
                                    this.size[1] = newSize[1];
                                } else {
                                    this.size = newSize;
                                }
                                this.setDirtyCanvas(true);
                            }
                        } else {
                            // Input disconnected - simple cleanup
                            this.cleanupEmptyInputs();
                        }
                    } catch (e) {
                        console.error("PromptManager: Error handling connection change:", e);
                    } finally {
                        this.isProcessing = false;
                    }
                }
            };
            
            // Simple cleanup method
            nodeType.prototype.cleanupEmptyInputs = function() {
                // Don't cleanup during processing or loading
                if (this.isProcessing || app.graph.isLoading) {
                    return;
                }
                
                // Find the highest connected input
                let highestConnected = -1;
                for (let i = 0; i < this.inputs.length; i++) {
                    if (this.inputs[i] && this.inputs[i].link !== null) {
                        highestConnected = i;
                    }
                }
                
                // Keep inputs up to highest connected + 1 spare
                const targetLength = Math.max(highestConnected + 2, 1);
                
                // Remove excess inputs (but keep at least one)
                while (this.inputs.length > targetLength && this.inputs.length > 1) {
                    this.removeInput(this.inputs.length - 1);
                }
            };
            
            // Simple execution handler
            const origOnExecute = nodeType.prototype.onExecute;
            nodeType.prototype.onExecute = function() {
                // Collect all connected inputs
                const collectedInputs = [];
                
                for (let i = 0; i < this.inputs.length; i++) {
                    if (this.inputs[i] && this.inputs[i].link !== null) {
                        const inputData = this.getInputData(i);
                        if (inputData !== undefined && inputData !== null) {
                            if (Array.isArray(inputData)) {
                                collectedInputs.push(...inputData);
                            } else {
                                collectedInputs.push(inputData);
                            }
                        }
                    }
                }
                
                // Store collected inputs temporarily
                this._collectedInputs = collectedInputs.length > 0 ? collectedInputs : null;
                
                if (origOnExecute) {
                    return origOnExecute.apply(this, arguments);
                }
            };
            
            // Safe serialization
            const origOnSerialize = nodeType.prototype.onSerialize;
            nodeType.prototype.onSerialize = function(o) {
                if (origOnSerialize) {
                    origOnSerialize.apply(this, arguments);
                }
                
                // Save the number of inputs
                o.dynamicInputCount = this.inputs ? this.inputs.length : 1;
            };
            
            // Safe configuration restore
            const origOnConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function(o) {
                // Clear any existing state
                this.isProcessing = false;
                this._collectedInputs = null;
                
                // Call original configure first to restore connections
                if (origOnConfigure) {
                    origOnConfigure.apply(this, arguments);
                }
                
                // After connections are restored, ensure we have the right number of inputs
                if (o.dynamicInputCount && o.dynamicInputCount > 1) {
                    // Only adjust if current count doesn't match saved count
                    const targetCount = Math.min(o.dynamicInputCount, this.maxInputs || 10);
                    
                    // Add missing inputs
                    while (this.inputs.length < targetCount) {
                        const index = this.inputs.length;
                        if (index === 0) {
                            this.addInput("context", "*");
                        } else {
                            this.addInput(`context_${index}`, "*");
                        }
                    }
                    
                    // Remove excess inputs (but preserve connections)
                    while (this.inputs.length > targetCount && this.inputs.length > 1) {
                        const lastInput = this.inputs[this.inputs.length - 1];
                        // Only remove if not connected
                        if (!lastInput || lastInput.link === null) {
                            this.removeInput(this.inputs.length - 1);
                        } else {
                            break; // Stop if we hit a connected input
                        }
                    }
                }
                
                // Ensure at least one input exists
                if (!this.inputs || this.inputs.length === 0) {
                    this.addInput("context", "*");
                }

                // After loading and configuring, cleanup and resize
                this.cleanupEmptyInputs();
                
                const newSize = this.computeSize();
                if (this.size) {
                    this.size[1] = newSize[1];
                } else {
                    this.size = newSize;
                }
                this.setDirtyCanvas(true, true);
            };
        }
    }
});