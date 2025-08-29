// Dynamic input ports for LLMPromptManager node
import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "LLMToolkit.PromptManagerDynamic",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "LLMPromptManager") {
            const origOnNodeCreated = nodeType.prototype.onNodeCreated;
            
            nodeType.prototype.onNodeCreated = function() {
                const ret = origOnNodeCreated ? origOnNodeCreated.apply(this, arguments) : undefined;
                
                // Store original input configuration
                this.originalInputs = this.inputs ? [...this.inputs] : [];
                
                // Initialize with just one context input
                this.inputs = [];
                this.addInput("context", "*");
                
                // Track connected inputs
                this.connectedInputs = new Set();
                this.maxInputIndex = 0;
                
                // Store accumulated data internally
                this.accumulatedData = [];
                
                return ret;
            };
            
            // Override onConnectionsChange to handle dynamic inputs
            const origOnConnectionsChange = nodeType.prototype.onConnectionsChange;
            nodeType.prototype.onConnectionsChange = function(type, index, connected, link_info) {
                if (origOnConnectionsChange) {
                    origOnConnectionsChange.apply(this, arguments);
                }
                
                if (type === 1) { // Input connection
                    if (connected) {
                        // Mark this input as connected
                        this.connectedInputs.add(index);
                        
                        // Check if this is the last available input
                        const lastInputIndex = this.inputs.length - 1;
                        if (index === lastInputIndex) {
                            // Add a new input port
                            const nextIndex = this.inputs.length;
                            this.addInput(`context_${nextIndex}`, "*");
                            this.maxInputIndex = nextIndex;
                            
                            // Update size to accommodate new input
                            if (this.size && this.size[1] < (80 + (this.inputs.length * 25))) {
                                this.size[1] = 80 + (this.inputs.length * 25);
                            }
                        }
                    } else {
                        // Input disconnected
                        this.connectedInputs.delete(index);
                        
                        // Clean up empty trailing inputs (keep at least one)
                        this.cleanupEmptyInputs();
                    }
                }
            };
            
            // Add method to clean up empty trailing inputs
            nodeType.prototype.cleanupEmptyInputs = function() {
                // Find the highest connected input index
                let highestConnected = -1;
                for (let i = 0; i < this.inputs.length; i++) {
                    if (this.inputs[i].link !== null) {
                        highestConnected = i;
                    }
                }
                
                // Remove inputs after the highest connected one, keeping at least one extra
                const targetLength = Math.max(highestConnected + 2, 1);
                while (this.inputs.length > targetLength && this.inputs.length > 1) {
                    this.removeInput(this.inputs.length - 1);
                }
                
                // Ensure we always have at least one unconnected input at the end
                const hasUnconnectedAtEnd = this.inputs.length === 0 || 
                    this.inputs[this.inputs.length - 1].link === null;
                    
                if (!hasUnconnectedAtEnd) {
                    const nextIndex = this.inputs.length;
                    this.addInput(`context_${nextIndex}`, "*");
                }
            };
            
            // Override the execution to collect all inputs
            const origOnExecute = nodeType.prototype.onExecute;
            nodeType.prototype.onExecute = function() {
                // Collect all connected inputs into an array
                const collectedInputs = [];
                
                for (let i = 0; i < this.inputs.length; i++) {
                    if (this.inputs[i].link !== null) {
                        const inputData = this.getInputData(i);
                        if (inputData !== undefined && inputData !== null) {
                            // If input is already an array/list, spread it
                            if (Array.isArray(inputData)) {
                                collectedInputs.push(...inputData);
                            } else {
                                collectedInputs.push(inputData);
                            }
                        }
                    }
                }
                
                // Store the collected inputs for the Python side
                // Pass as a list to the first input
                if (collectedInputs.length > 0) {
                    // Create a unified context that the Python side will process
                    this.widgets_values = this.widgets_values || [];
                    
                    // The Python node will receive this as the context parameter
                    // We'll modify the inputs to be passed as a list
                    this.inputData = collectedInputs;
                }
                
                if (origOnExecute) {
                    return origOnExecute.apply(this, arguments);
                }
            };
            
            // Override getInputData to return our collected data
            const origGetInputData = nodeType.prototype.getInputData;
            nodeType.prototype.getInputData = function(slot) {
                // For the first slot (slot 0), return all collected inputs as a list
                if (slot === 0 && this.inputData && this.inputData.length > 0) {
                    return this.inputData;
                }
                
                // For other slots, return the original data
                if (origGetInputData) {
                    return origGetInputData.apply(this, arguments);
                }
                
                return LiteGraph.prototype.getInputData.apply(this, arguments);
            };
            
            // Override onSerialize to save the dynamic inputs
            const origOnSerialize = nodeType.prototype.onSerialize;
            nodeType.prototype.onSerialize = function(o) {
                if (origOnSerialize) {
                    origOnSerialize.apply(this, arguments);
                }
                
                // Save the number of inputs
                o.dynamicInputCount = this.inputs ? this.inputs.length : 1;
            };
            
            // Override onConfigure to restore dynamic inputs
            const origOnConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function(o) {
                // Restore the correct number of inputs
                if (o.dynamicInputCount && o.dynamicInputCount > 1) {
                    // Clear existing inputs
                    while (this.inputs && this.inputs.length > 0) {
                        this.removeInput(this.inputs.length - 1);
                    }
                    
                    // Add the saved number of inputs
                    for (let i = 0; i < o.dynamicInputCount; i++) {
                        if (i === 0) {
                            this.addInput("context", "*");
                        } else {
                            this.addInput(`context_${i}`, "*");
                        }
                    }
                }
                
                if (origOnConfigure) {
                    origOnConfigure.apply(this, arguments);
                }
                
                // Clean up any excess empty inputs after loading
                setTimeout(() => {
                    this.cleanupEmptyInputs();
                }, 100);
            };
        }
    }
});