import WebGPUResourceGroup from "./WebGPUResourceGroup";

export type WebGPUContextOptions = {
    device?: GPUDevice;
    powerPreference?: GPUPowerPreference;
    requiredFeatures?: Iterable<GPUFeatureName>;
};

class WebGPUContext {
    device?: GPUDevice;
    adapterInfo?: GPUAdapterInfo;
    options?: WebGPUContextOptions;

    constructor(options?: WebGPUContextOptions) {
        this.device = options?.device;
        this.options = options;
    }

    async initialize() {
        if (!navigator.gpu) {
            throw new Error("WebGPUContext: WebGPU is not supported");
        }

        const adapter = await navigator.gpu.requestAdapter({
            powerPreference: this.options?.powerPreference,
        });

        if (!adapter) {
            throw new Error("WebGPUContext: no GPUAdapters found");
        }

        this.adapterInfo = await adapter.requestAdapterInfo();

        this.device = await adapter.requestDevice({
            requiredFeatures: this.options?.requiredFeatures,
            requiredLimits: {
                maxStorageBuffersPerShaderStage: 10,
            },
        });

        console.log(
            "WebGPUContext: initialized with GPUAdapter",
            this.adapterInfo,
        );
    }

    createResourceGroup(): WebGPUResourceGroup {
        return new WebGPUResourceGroup(this);
    }
}

export default WebGPUContext;
