import { app } from "/scripts/app.js";

// Resolution configuration matching the Python code
const RESOLUTIONS = {
    "I2V720p": {
        "Horizontal": {"HQ": [1280, 720], "MQ": [832, 480], "LQ": [704, 544]},
        "Vertical": {"HQ": [720, 1280], "MQ": [480, 832], "LQ": [544, 704]},
        "Squarish": {"HQ": [624, 624], "MQ": [624, 624], "LQ": [624, 624]},
    },
    "I2V480p": {
        "Horizontal": {"HQ": [832, 480], "MQ": [704, 544], "LQ": [704, 544]},
        "Vertical": {"HQ": [480, 832], "MQ": [544, 704], "LQ": [544, 704]},
        "Squarish": {"HQ": [624, 624], "MQ": [624, 624], "LQ": [624, 624]},
    },
    "T2V14B": {
        "Horizontal": {"HQ": [1280, 720], "MQ": [1088, 832], "LQ": [832, 480]},
        "Vertical": {"HQ": [720, 1280], "MQ": [832, 1088], "LQ": [480, 832]},
        "Squarish": {"HQ": [960, 960], "MQ": [624, 624], "LQ": [544, 704]},
    },
    "T2V1.3B": {
        "Horizontal": {"HQ": [832, 480], "MQ": [704, 544], "LQ": [704, 544]},
        "Vertical": {"HQ": [480, 832], "MQ": [544, 704], "LQ": [544, 704]},
        "Squarish": {"HQ": [624, 624], "MQ": [624, 624], "LQ": [624, 624]},
    },
    "IMG": {
        "Horizontal": {"HQ": [1600, 900], "MQ": [1280, 720], "LQ": [1024, 576]},
        "Vertical": {"HQ": [900, 1600], "MQ": [720, 1280], "LQ": [576, 1024]},
        "Squarish": {"HQ": [1600, 1600], "MQ": [1024, 1024], "LQ": [512, 512]},
        "Cinematic": {"HQ": [1600, 688], "MQ": [1280, 550], "LQ": [1024, 440]},
    },
    "KONTEXT": {
        "Vertical": {"HQ": [672, 1568], "MQ": [720, 1456], "LQ": [832, 1248]},
        "Horizontal": {"HQ": [1568, 672], "MQ": [1456, 720], "LQ": [1248, 832]},
        "Squarish": {"HQ": [1024, 1024], "MQ": [944, 1104], "LQ": [880, 1184]},
    },
    "QWEN": {
        "Square": {"HQ": [1024, 1024], "MQ": [768, 768], "LQ": [512, 512]},
        "Landscape": {"HQ": [1280, 720], "MQ": [1024, 768], "LQ": [832, 624]},
        "Portrait": {"HQ": [720, 1280], "MQ": [768, 1024], "LQ": [624, 832]},
        "Wide": {"HQ": [1536, 768], "MQ": [1280, 640], "LQ": [1024, 512]},
        "Tall": {"HQ": [768, 1536], "MQ": [640, 1280], "LQ": [512, 1024]},
        "UltraWide": {"HQ": [1792, 768], "MQ": [1536, 640], "LQ": [1280, 544]},
        "UltraTall": {"HQ": [768, 1792], "MQ": [640, 1536], "LQ": [544, 1280]},
    },
    "GPT_IMAGE_1": {
        "Square":   {"HQ": [1024, 1024], "MQ": [1024, 1024], "LQ": [1024, 1024]},
        "Portrait": {"HQ": [1024, 1536], "MQ": [1024, 1536], "LQ": [1024, 1536]},
        "Landscape":{"HQ": [1536, 1024], "MQ": [1536, 1024], "LQ": [1536, 1024]},
    },
    "GEMINI_IMAGEN": {
        "Square (1:1)":      {"HQ": [1024, 1024], "MQ": [1024, 1024], "LQ": [1024, 1024]},
        "Portrait (3:4)":    {"HQ": [896, 1200],  "MQ": [768, 1024],  "LQ": [672, 896]},
        "Landscape (4:3)":   {"HQ": [1200, 896],  "MQ": [1024, 768],  "LQ": [896, 672]},
        "Portrait (9:16)":   {"HQ": [864, 1536],  "MQ": [720, 1280], "LQ": [576, 1024]},
        "Landscape (16:9)":  {"HQ": [1536, 864],  "MQ": [1280, 720], "LQ": [1024, 576]},
    },
    "BFL": {
        "1:1":   {"HQ": [1024, 1024], "MQ": [768, 768], "LQ": [512, 512]},
        "3:4":   {"HQ": [768, 1024], "MQ": [512, 682], "LQ": [384, 512]},
        "4:3":   {"HQ": [1024, 768], "MQ": [682, 512], "LQ": [512, 384]},
        "9:16":  {"HQ": [720, 1280], "MQ": [576, 1024], "LQ": [405, 720]},
        "16:9":  {"HQ": [1280, 720], "MQ": [1024, 576], "LQ": [720, 405]},
        "21:9":  {"HQ": [1536, 658], "MQ": [1280, 548], "LQ": [1024, 438]},
        "9:21":  {"HQ": [658, 1536], "MQ": [548, 1280], "LQ": [438, 1024]},
    },
    "SEEDREAM_V4": {
        "Square (1:1)":      {"HQ": [2048, 2048], "MQ": [1536, 1536], "LQ": [1024, 1024]},
        "Landscape (16:9)":  {"HQ": [2048, 1152], "MQ": [1536, 864], "LQ": [1024, 576]},
        "Portrait (9:16)":   {"HQ": [1152, 2048], "MQ": [864, 1536], "LQ": [576, 1024]},
        "Landscape (4:3)":   {"HQ": [2048, 1536], "MQ": [1536, 1152], "LQ": [1024, 768]},
        "Portrait (3:4)":    {"HQ": [1536, 2048], "MQ": [1152, 1536], "LQ": [768, 1024]},
    },
    "HUNYUAN": {
        "Landscape (16:9)": {"HQ": [2560, 1536], "MQ": [2560, 1536], "LQ": [2560, 1536]},
        "Landscape (4:3)":  {"HQ": [2304, 1792], "MQ": [2304, 1792], "LQ": [2304, 1792]},
        "Square (1:1)":     {"HQ": [2048, 2048], "MQ": [2048, 2048], "LQ": [2048, 2048]},
        "Portrait (3:4)":   {"HQ": [1792, 2304], "MQ": [1792, 2304], "LQ": [1792, 2304]},
        "Portrait (9:16)":  {"HQ": [1536, 2560], "MQ": [1536, 2560], "LQ": [1536, 2560]},
    },
};
RESOLUTIONS["NANO_BANANA"] = RESOLUTIONS["GEMINI_IMAGEN"];
RESOLUTIONS["FLUX_DEV"] = RESOLUTIONS["IMG"];

app.registerExtension({
    name: "ComfyLLMToolkit.ResolutionSelector",
    
    async nodeCreated(node) {
        if (node.comfyClass === "ResolutionSelector") {
            const modeWidget = node.widgets.find(w => w.name === "mode");
            const aspectWidget = node.widgets.find(w => w.name === "aspect_ratio");

            if (!modeWidget || !aspectWidget) return;

            const updateAspectRatios = (mode) => {
                const aspects = Object.keys(RESOLUTIONS[mode] || {});
                aspectWidget.options.values = aspects;
                if (!aspects.includes(aspectWidget.value)) {
                    aspectWidget.value = aspects[0];
                }
            };

            const originalModeCallback = modeWidget.callback;
            modeWidget.callback = function() {
                if (originalModeCallback) {
                    originalModeCallback.apply(this, arguments);
                }
                updateAspectRatios(this.value);
            };

            // Initial setup
            setTimeout(() => updateAspectRatios(modeWidget.value), 0);
        }
    },
    
    async loadedGraphNode(node) {
        if (node.comfyClass === "ResolutionSelector") {
            const modeWidget = node.widgets.find(w => w.name === "mode");
            const aspectWidget = node.widgets.find(w => w.name === "aspect_ratio");
            if (modeWidget && aspectWidget) {
                const aspects = Object.keys(RESOLUTIONS[modeWidget.value] || {});
                aspectWidget.options.values = aspects;
                if (!aspects.includes(aspectWidget.value)) {
                    aspectWidget.value = aspects[0];
                }
            }
        }
    }
});