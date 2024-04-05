import WebGPUContext from "./WebGPUContext";
import { pathTraceShader } from "./shaders/pathTrace";

class WebGPUPathTrace {
    context: WebGPUContext;
    numBounces: number;
    debug: boolean;
    writeDepth: boolean;
    workgroupSizeX: number;
    workgroupSizeY: number;

    pathTracePipeline?: GPUComputePipeline;

    constructor(
        context: WebGPUContext,
        numBounces: number,
        debug: boolean = false,
        writeDepth: boolean = false,
        workgroupSizeX: number = 16,
        workgroupSizeY: number = 16,
    ) {
        this.context = context;
        this.numBounces = numBounces;
        this.debug = debug;
        this.writeDepth = writeDepth;
        this.workgroupSizeX = workgroupSizeX;
        this.workgroupSizeY = workgroupSizeY;
    }

    async initialize() {
        if (!this.context.device) {
            this.context.initialize();
        }

        const pathTraceModule = this.context.device!.createShaderModule({
            code: pathTraceShader(
                this.workgroupSizeX,
                this.workgroupSizeY,
                this.numBounces,
                this.debug,
                this.writeDepth,
            ),
        });

        const pathTraceBindGroupLayouts: GPUBindGroupLayout[] = [];

        pathTraceBindGroupLayouts.push(
            this.context.device!.createBindGroupLayout({
                entries: [
                    {
                        binding: 0,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: {
                            type: "read-only-storage",
                        },
                    },
                    {
                        binding: 1,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: {
                            type: "read-only-storage",
                        },
                    },
                    {
                        binding: 2,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: {
                            type: "read-only-storage",
                        },
                    },
                    {
                        binding: 3,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: {
                            type: "read-only-storage",
                        },
                    },
                    {
                        binding: 4,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: {
                            type: "read-only-storage",
                        },
                    },
                    {
                        binding: 5,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: {
                            type: "read-only-storage",
                        },
                    },
                    {
                        binding: 6,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: {
                            type: "read-only-storage",
                        },
                    },
                    {
                        binding: 7,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: {
                            type: "read-only-storage",
                        },
                    },
                    {
                        binding: 8,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: {
                            type: "uniform",
                        },
                    },
                    {
                        binding: 9,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: {
                            type: "uniform",
                        },
                    },
                    {
                        binding: 10,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: {
                            type: "storage",
                        },
                    },
                    {
                        binding: 11,
                        visibility: GPUShaderStage.COMPUTE,
                        storageTexture: {
                            format: "rgba8unorm",
                        },
                    },
                ],
            }),
        );

        if (this.debug) {
            pathTraceBindGroupLayouts.push(
                this.context.device!.createBindGroupLayout({
                    entries: [
                        {
                            binding: 0,
                            visibility: GPUShaderStage.COMPUTE,
                            buffer: {
                                type: "storage",
                            },
                        },
                    ],
                }),
            );
        }

        if (this.writeDepth) {
            pathTraceBindGroupLayouts.push(
                this.context.device!.createBindGroupLayout({
                    entries: [
                        {
                            binding: 0,
                            visibility: GPUShaderStage.COMPUTE,
                            storageTexture: {
                                format: "r32float",
                            },
                        },
                    ],
                }),
            );
        }

        this.pathTracePipeline = this.context.device!.createComputePipeline({
            layout: this.context.device!.createPipelineLayout({
                bindGroupLayouts: pathTraceBindGroupLayouts,
            }),
            compute: {
                module: pathTraceModule,
                entryPoint: "main",
            },
        });
    }

    encodePathTracePass(
        passEncoder: GPUComputePassEncoder,
        objectsBuffer: GPUBuffer,
        lightsBuffer: GPUBuffer,
        positionsBuffer: GPUBuffer,
        normalsBuffer: GPUBuffer,
        indexBuffer: GPUBuffer,
        triangleToObjectBuffer: GPUBuffer,
        cameraBuffer: GPUBuffer,
        pathTraceDataBuffer: GPUBuffer,
        cumulativeBuffer: GPUBuffer,
        canvasTextureView: GPUTextureView,
        canvasDimensionX: number,
        canvasDimensionY: number,
        bvhNodesBuffer: GPUBuffer,
        bvhNodeBoundsBuffer: GPUBuffer,
        debugBuffer?: GPUBuffer,
        depthTextureView?: GPUTextureView,
    ) {
        if (!this.context.device) {
            throw new Error("WebGPUPathTrace: WebGPUContext not initialized");
        }

        const pathTraceBindGroup = this.context.device.createBindGroup({
            layout: this.pathTracePipeline!.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: objectsBuffer,
                    },
                },
                {
                    binding: 1,
                    resource: {
                        buffer: lightsBuffer,
                    },
                },
                {
                    binding: 2,
                    resource: {
                        buffer: positionsBuffer,
                    },
                },
                {
                    binding: 3,
                    resource: {
                        buffer: normalsBuffer,
                    },
                },
                {
                    binding: 4,
                    resource: {
                        buffer: indexBuffer,
                    },
                },
                {
                    binding: 5,
                    resource: {
                        buffer: triangleToObjectBuffer,
                    },
                },
                {
                    binding: 6,
                    resource: {
                        buffer: bvhNodesBuffer,
                    },
                },
                {
                    binding: 7,
                    resource: {
                        buffer: bvhNodeBoundsBuffer,
                    },
                },
                {
                    binding: 8,
                    resource: {
                        buffer: cameraBuffer,
                    },
                },
                {
                    binding: 9,
                    resource: {
                        buffer: pathTraceDataBuffer,
                    },
                },
                {
                    binding: 10,
                    resource: {
                        buffer: cumulativeBuffer,
                    },
                },
                {
                    binding: 11,
                    resource: canvasTextureView,
                },
            ],
        });

        passEncoder.setPipeline(this.pathTracePipeline!);
        passEncoder.setBindGroup(0, pathTraceBindGroup);

        let currentBindGroupNo = 1;

        if (this.debug) {
            var pathTraceDebugBindGroup = this.context.device.createBindGroup({
                layout: this.pathTracePipeline!.getBindGroupLayout(
                    currentBindGroupNo,
                ),
                entries: [
                    {
                        binding: 0,
                        resource: {
                            buffer: debugBuffer!,
                        },
                    },
                ],
            });

            passEncoder.setBindGroup(
                currentBindGroupNo++,
                pathTraceDebugBindGroup,
            );
        }

        if (this.writeDepth) {
            var pathTraceDepthBindGroup = this.context.device.createBindGroup({
                layout: this.pathTracePipeline!.getBindGroupLayout(
                    currentBindGroupNo,
                ),
                entries: [
                    {
                        binding: 0,
                        resource: depthTextureView!,
                    },
                ],
            });

            passEncoder.setBindGroup(
                currentBindGroupNo++,
                pathTraceDepthBindGroup,
            );
        }

        const numWorkgroupsX = Math.ceil(
            canvasDimensionX / this.workgroupSizeX,
        );

        const numWorkgroupsY = Math.ceil(
            canvasDimensionY / this.workgroupSizeY,
        );

        passEncoder.dispatchWorkgroups(numWorkgroupsX, numWorkgroupsY);
    }
}

export default WebGPUPathTrace;
