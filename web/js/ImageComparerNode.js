import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

function imageDataToUrl(data) {
    return api.apiURL(`/view?filename=${encodeURIComponent(data.filename)}&type=${data.type}&subfolder=${data.subfolder}${app.getPreviewFormatParam()}${app.getRandParam()}`);
}

class ImageComparerWidget {
    constructor(name, node) {
        this.name = name;
        this.type = 'custom';
        this.hitAreas = {};
        this.selected = [];
        this._value = { images: [] };
        this.node = node;
    }

    set value(v) {
        let cleanedVal;
        if (Array.isArray(v)) {
            cleanedVal = v.map((d, i) => {
                if (!d || typeof d === "string") {
                    d = { url: d, name: i == 0 ? "A" : "B", selected: true };
                }
                return d;
            });
        } else {
            cleanedVal = v.images || [];
        }

        if (cleanedVal.length > 2) {
            const hasAAndB = cleanedVal.some((i) => i.name.startsWith("A")) &&
                cleanedVal.some((i) => i.name.startsWith("B"));
            if (!hasAAndB) {
                cleanedVal = [cleanedVal[0], cleanedVal[1]];
            }
        }

        let selected = cleanedVal.filter((d) => d.selected);
        if (!selected.length && cleanedVal.length) {
            cleanedVal[0].selected = true;
        }

        selected = cleanedVal.filter((d) => d.selected);
        if (selected.length === 1 && cleanedVal.length > 1) {
            cleanedVal.find((d) => !d.selected).selected = true;
        }

        this._value.images = cleanedVal;
        selected = cleanedVal.filter((d) => d.selected);
        this.setSelected(selected);
    }

    get value() {
        return this._value;
    }

    setSelected(selected) {
        this._value.images.forEach((d) => (d.selected = false));
        this.node.imgs.length = 0;
        
        for (const sel of selected) {
            if (!sel.img) {
                sel.img = new Image();
                sel.img.src = sel.url;
                sel.img.onload = () => {
                    this.node.setDirtyCanvas(true);
                };
                this.node.imgs.push(sel.img);
            }
            sel.selected = true;
        }
        this.selected = selected;
    }

    draw(ctx, node, width, y) {
        this.hitAreas = {};
        
        // Draw selector buttons if multiple images
        if (this.value.images.length > 2) {
            ctx.textAlign = "left";
            ctx.textBaseline = "top";
            ctx.font = `14px Arial`;
            const drawData = [];
            const spacing = 5;
            let x = 0;
            
            for (const img of this.value.images) {
                const textWidth = ctx.measureText(img.name).width;
                drawData.push({
                    img,
                    text: img.name,
                    x,
                    width: textWidth,
                });
                x += textWidth + spacing;
            }
            
            x = (node.size[0] - (x - spacing)) / 2;
            for (const d of drawData) {
                ctx.fillStyle = d.img.selected ? "rgba(180, 180, 180, 1)" : "rgba(180, 180, 180, 0.5)";
                ctx.fillText(d.text, x, y);
                this.hitAreas[d.text] = {
                    bounds: [x, y, d.width, 14],
                    data: d.img,
                    onDown: this.onSelectionDown.bind(this),
                };
                x += d.width + spacing;
            }
            y += 20;
        }

        // Draw images
        if (node.properties?.["comparer_mode"] === "Click") {
            this.drawImage(ctx, this.selected[node.isPointerDown ? 1 : 0], node, y);
        } else {
            // Slide mode - draw base image
            this.drawImage(ctx, this.selected[0], node, y);
            // Draw overlay image if hovering
            if (node.isPointerOver && this.selected[1]) {
                this.drawImage(ctx, this.selected[1], node, y, node.pointerOverPos[0]);
            }
        }
    }

    onSelectionDown(event, pos, node, bounds) {
        const selected = [...this.selected];
        if (bounds?.data.name.startsWith("A")) {
            selected[0] = bounds.data;
        } else if (bounds?.data.name.startsWith("B")) {
            selected[1] = bounds.data;
        }
        this.setSelected(selected);
        node.setDirtyCanvas(true);
    }

    drawImage(ctx, image, node, y, cropX) {
        if (!image?.img?.naturalWidth || !image?.img?.naturalHeight) {
            return;
        }

        let [nodeWidth, nodeHeight] = node.size;
        const imageAspect = image.img.naturalWidth / image.img.naturalHeight;
        let height = nodeHeight - y;
        const widgetAspect = nodeWidth / height;
        let targetWidth, targetHeight;
        let offsetX = 0;

        if (imageAspect > widgetAspect) {
            targetWidth = nodeWidth;
            targetHeight = nodeWidth / imageAspect;
        } else {
            targetHeight = height;
            targetWidth = height * imageAspect;
            offsetX = (nodeWidth - targetWidth) / 2;
        }

        const widthMultiplier = image.img.naturalWidth / targetWidth;
        const sourceX = 0;
        const sourceY = 0;
        const sourceWidth = cropX != null ? (cropX - offsetX) * widthMultiplier : image.img.naturalWidth;
        const sourceHeight = image.img.naturalHeight;
        const destX = (nodeWidth - targetWidth) / 2;
        const destY = y + (height - targetHeight) / 2;
        const destWidth = cropX != null ? cropX - offsetX : targetWidth;
        const destHeight = targetHeight;

        ctx.save();
        ctx.beginPath();

        if (cropX) {
            ctx.rect(destX, destY, destWidth, destHeight);
            ctx.clip();
        }

        ctx.drawImage(image.img, sourceX, sourceY, sourceWidth, sourceHeight, destX, destY, destWidth, destHeight);

        if (cropX != null && cropX >= destX && cropX <= destX + targetWidth) {
            ctx.beginPath();
            ctx.moveTo(cropX, destY);
            ctx.lineTo(cropX, destY + destHeight);
            ctx.globalCompositeOperation = "difference";
            ctx.strokeStyle = "rgba(255,255,255, 1)";
            ctx.lineWidth = 2;
            ctx.stroke();
        }

        ctx.restore();
    }

    computeSize(width) {
        return [width, Math.max(200, width * 0.75)];
    }

    serializeValue() {
        const v = [];
        for (const data of this._value.images) {
            const d = { ...data };
            delete d.img;
            v.push(d);
        }
        return { images: v };
    }
}

app.registerExtension({
    name: "llmtoolkit.ImageComparer",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "ImageComparer") {
            const origOnExecuted = nodeType.prototype.onExecuted;
            
            nodeType.prototype.onNodeCreated = function() {
                this.imageIndex = 0;
                this.imgs = [];
                this.serialize_widgets = true;
                this.isPointerDown = false;
                this.isPointerOver = false;
                this.pointerOverPos = [0, 0];
                this.properties = this.properties || {};
                this.properties["comparer_mode"] = "Slide";
                
                this.canvasWidget = this.addCustomWidget(new ImageComparerWidget("llmtoolkit_comparer", this));
                this.setSize(this.computeSize());
                this.setDirtyCanvas(true, true);
            };

            nodeType.prototype.onExecuted = function(output) {
                if (origOnExecuted) {
                    origOnExecuted.apply(this, arguments);
                }
                
                if (!this.canvasWidget) return;
                
                if ("images" in output) {
                    this.canvasWidget.value = {
                        images: (output.images || []).map((d, i) => {
                            return {
                                name: i === 0 ? "A" : "B",
                                selected: true,
                                url: imageDataToUrl(d),
                            };
                        }),
                    };
                } else {
                    output.a_images = output.a_images || [];
                    output.b_images = output.b_images || [];
                    const imagesToChoose = [];
                    const multiple = output.a_images.length + output.b_images.length > 2;
                    
                    for (const [i, d] of output.a_images.entries()) {
                        imagesToChoose.push({
                            name: output.a_images.length > 1 || multiple ? `A${i + 1}` : "A",
                            selected: i === 0,
                            url: imageDataToUrl(d),
                        });
                    }
                    
                    for (const [i, d] of output.b_images.entries()) {
                        imagesToChoose.push({
                            name: output.b_images.length > 1 || multiple ? `B${i + 1}` : "B",
                            selected: i === 0,
                            url: imageDataToUrl(d),
                        });
                    }
                    
                    this.canvasWidget.value = { images: imagesToChoose };
                }
                
                this.setDirtyCanvas(true);
            };

            nodeType.prototype.setIsPointerDown = function(down = this.isPointerDown) {
                const newIsDown = down && !!app.canvas.pointer_is_down;
                if (this.isPointerDown !== newIsDown) {
                    this.isPointerDown = newIsDown;
                    this.setDirtyCanvas(true, false);
                }
                this.imageIndex = this.isPointerDown ? 1 : 0;
                if (this.isPointerDown) {
                    requestAnimationFrame(() => {
                        this.setIsPointerDown();
                    });
                }
            };

            nodeType.prototype.onMouseDown = function(event, pos, canvas) {
                this.setIsPointerDown(true);
                return false;
            };

            nodeType.prototype.onMouseEnter = function(event) {
                this.setIsPointerDown(!!app.canvas.pointer_is_down);
                this.isPointerOver = true;
            };

            nodeType.prototype.onMouseLeave = function(event) {
                this.setIsPointerDown(false);
                this.isPointerOver = false;
            };

            nodeType.prototype.onMouseMove = function(event, pos, canvas) {
                this.pointerOverPos = [...pos];
                this.imageIndex = this.pointerOverPos[0] > this.size[0] / 2 ? 1 : 0;
                this.setDirtyCanvas(true, false);
            };

            // Add property configuration
            nodeType["@comparer_mode"] = {
                type: "combo",
                values: ["Slide", "Click"],
            };
        }
    },
});