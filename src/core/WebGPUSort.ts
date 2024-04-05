import WebGPUContext from "./WebGPUContext";
import WebGPUResourceGroup from "./WebGPUResourceGroup";
import {
    scanShader,
    recurseShader,
    countShader,
    scatterShader,
} from "./shaders/sort";

class WebGPUSort {
    context: WebGPUContext;
    workgroupSize: number;
    workgroupSizeDouble: number;

    scanWithoutRecursePipeline?: GPUComputePipeline;
    scanWithRecursePipeline?: GPUComputePipeline;
    recursePipeline?: GPUComputePipeline;
    countsPipelines?: GPUComputePipeline[];
    scatterPipelines?: GPUComputePipeline[];

    sumsBuffers?: GPUBuffer[];
    countsBuffer?: GPUBuffer;
    outputBuffer?: GPUBuffer;
    outputIdsBuffer?: GPUBuffer;

    constructor(context: WebGPUContext, workgroupSize = 128) {
        this.context = context;
        this.workgroupSize = workgroupSize;
        this.workgroupSizeDouble = workgroupSize << 1;
    }

    async initialize() {
        if (!this.context.device) {
            await this.context.initialize();
        }

        const scanWithoutRecurseModule =
            this.context.device!.createShaderModule({
                code: scanShader(this.workgroupSize, false),
            });

        const scanWithRecurseModule = this.context.device!.createShaderModule({
            code: scanShader(this.workgroupSize, true),
        });

        const scanWithoutRecurseBindGroupLayout =
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
            });

        const scanWithRecurseBindGroupLayout =
            this.context.device!.createBindGroupLayout({
                entries: [
                    {
                        binding: 0,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: {
                            type: "storage",
                        },
                    },
                    {
                        binding: 1,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: {
                            type: "storage",
                        },
                    },
                ],
            });

        this.scanWithoutRecursePipeline =
            this.context.device!.createComputePipeline({
                layout: this.context.device!.createPipelineLayout({
                    bindGroupLayouts: [scanWithoutRecurseBindGroupLayout],
                }),
                compute: {
                    module: scanWithoutRecurseModule,
                    entryPoint: "main",
                },
            });

        this.scanWithRecursePipeline =
            this.context.device!.createComputePipeline({
                layout: this.context.device!.createPipelineLayout({
                    bindGroupLayouts: [scanWithRecurseBindGroupLayout],
                }),
                compute: {
                    module: scanWithRecurseModule,
                    entryPoint: "main",
                },
            });

        const recurseModule = this.context.device!.createShaderModule({
            code: recurseShader(this.workgroupSize),
        });

        const recurseBindGroupLayout =
            this.context.device!.createBindGroupLayout({
                entries: [
                    {
                        binding: 0,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: {
                            type: "storage",
                        },
                    },
                    {
                        binding: 1,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: {
                            type: "read-only-storage",
                        },
                    },
                ],
            });

        this.recursePipeline = this.context.device!.createComputePipeline({
            layout: this.context.device!.createPipelineLayout({
                bindGroupLayouts: [recurseBindGroupLayout],
            }),
            compute: {
                module: recurseModule,
                entryPoint: "main",
            },
        });

        const countModule = this.context.device!.createShaderModule({
            code: countShader(this.workgroupSize),
        });

        const countsBindGroupLayout =
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
                            type: "storage",
                        },
                    },
                ],
            });

        const scatterModule = this.context.device!.createShaderModule({
            code: scatterShader(this.workgroupSize),
        });

        const scatterBindGroupLayout =
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
                            type: "storage",
                        },
                    },
                    {
                        binding: 3,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: {
                            type: "storage",
                        },
                    },
                    {
                        binding: 4,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: {
                            type: "storage",
                        },
                    },
                ],
            });

        this.countsPipelines = new Array(8);
        this.scatterPipelines = new Array(8);

        for (let iter = 0; iter < 8; iter++) {
            this.countsPipelines[iter] =
                this.context.device!.createComputePipeline({
                    layout: this.context.device!.createPipelineLayout({
                        bindGroupLayouts: [countsBindGroupLayout],
                    }),
                    compute: {
                        module: countModule,
                        entryPoint: "main",
                        constants: {
                            iter,
                        },
                    },
                });

            this.scatterPipelines[iter] =
                this.context.device!.createComputePipeline({
                    layout: this.context.device!.createPipelineLayout({
                        bindGroupLayouts: [scatterBindGroupLayout],
                    }),
                    compute: {
                        module: scatterModule,
                        entryPoint: "main",
                        constants: {
                            iter,
                        },
                    },
                });
        }
    }

    allocate(resourceGroup: WebGPUResourceGroup, maxInputLength: number) {
        this.sumsBuffers = [];
        let currentSumsLength = maxInputLength;

        do {
            currentSumsLength = Math.ceil(
                currentSumsLength / this.workgroupSizeDouble,
            );

            this.sumsBuffers.push(
                resourceGroup.createBuffer({
                    size: currentSumsLength * 4,
                    usage: GPUBufferUsage.STORAGE,
                }),
            );
        } while (currentSumsLength > this.workgroupSizeDouble);

        const countsBufferSize =
            Math.ceil(maxInputLength / this.workgroupSize) * 64;

        this.countsBuffer = resourceGroup.createBuffer({
            size: countsBufferSize,
            usage: GPUBufferUsage.STORAGE,
        });

        this.outputBuffer = resourceGroup.createBuffer({
            size: maxInputLength * 4,
            usage: GPUBufferUsage.STORAGE,
        });

        this.outputIdsBuffer = resourceGroup.createBuffer({
            size: maxInputLength * 4,
            usage: GPUBufferUsage.STORAGE,
        });
    }

    private encodeScanPass(
        passEncoder: GPUComputePassEncoder,
        inputBuffer: GPUBuffer,
        inputSize: number,
        depth: number = 0,
    ) {
        if (!this.context.device) {
            throw new Error("WebGPUSort: WebGPUContext not initialized");
        }

        const inputLength = inputSize / 4;
        const recurseNeeded = inputLength > this.workgroupSizeDouble;
        const numWorkgroups = Math.ceil(inputLength / this.workgroupSizeDouble);

        if (recurseNeeded) {
            var sumsSize = numWorkgroups * 4;
        }

        const scanPipeline = !recurseNeeded
            ? this.scanWithoutRecursePipeline!
            : this.scanWithRecursePipeline!;

        const scanBindGroup = this.context.device.createBindGroup({
            layout: scanPipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: inputBuffer,
                    },
                },
                ...(recurseNeeded
                    ? [
                          {
                              binding: 1,
                              resource: {
                                  buffer: this.sumsBuffers![depth],
                                  size: sumsSize!,
                              },
                          },
                      ]
                    : []),
            ],
        });

        passEncoder.setPipeline(scanPipeline);
        passEncoder.setBindGroup(0, scanBindGroup);
        passEncoder.dispatchWorkgroups(numWorkgroups);

        if (recurseNeeded) {
            this.encodeScanPass(
                passEncoder,
                this.sumsBuffers![depth],
                sumsSize!,
                depth + 1,
            );

            const recurseBindGroup = this.context.device.createBindGroup({
                layout: this.recursePipeline!.getBindGroupLayout(0),
                entries: [
                    {
                        binding: 0,
                        resource: {
                            buffer: inputBuffer,
                            size: inputSize,
                        },
                    },
                    {
                        binding: 1,
                        resource: {
                            buffer: this.sumsBuffers![depth],
                            size: sumsSize!,
                        },
                    },
                ],
            });

            passEncoder.setPipeline(this.recursePipeline!);
            passEncoder.setBindGroup(0, recurseBindGroup);
            passEncoder.dispatchWorkgroups(numWorkgroups - 1);
        }
    }

    encodeSortPass(
        passEncoder: GPUComputePassEncoder,
        inputBuffer: GPUBuffer,
        inputSize: number,
        inputIdsBuffer: GPUBuffer,
    ) {
        if (!this.context.device) {
            throw new Error("WebGPUSort: WebGPUContext not initialized");
        }

        const inputLength = inputSize / 4;
        const numWorkgroups = Math.ceil(inputLength / this.workgroupSize);
        const countsSize = numWorkgroups * 64;

        for (let iter = 0; iter < 8; iter++) {
            const countsBindGroup = this.context.device.createBindGroup({
                layout: this.countsPipelines![iter].getBindGroupLayout(0),
                entries: [
                    {
                        binding: 0,
                        resource: {
                            buffer: !(iter & 1)
                                ? inputBuffer
                                : this.outputBuffer!,
                            size: inputSize,
                        },
                    },
                    {
                        binding: 1,
                        resource: {
                            buffer: this.countsBuffer!,
                            size: countsSize,
                        },
                    },
                ],
            });

            passEncoder.setPipeline(this.countsPipelines![iter]);
            passEncoder.setBindGroup(0, countsBindGroup);
            passEncoder.dispatchWorkgroups(numWorkgroups);

            this.encodeScanPass(passEncoder, this.countsBuffer!, countsSize);

            const scatterBindGroup = this.context.device.createBindGroup({
                layout: this.scatterPipelines![iter].getBindGroupLayout(0),
                entries: [
                    {
                        binding: 0,
                        resource: {
                            buffer: !(iter & 1)
                                ? inputBuffer
                                : this.outputBuffer!,
                            size: inputSize,
                        },
                    },
                    {
                        binding: 1,
                        resource: {
                            buffer: !(iter & 1)
                                ? inputIdsBuffer
                                : this.outputIdsBuffer!,
                            size: inputSize,
                        },
                    },
                    {
                        binding: 2,
                        resource: {
                            buffer: iter & 1 ? inputBuffer : this.outputBuffer!,
                            size: inputSize,
                        },
                    },
                    {
                        binding: 3,
                        resource: {
                            buffer:
                                iter & 1
                                    ? inputIdsBuffer
                                    : this.outputIdsBuffer!,
                            size: inputSize,
                        },
                    },
                    {
                        binding: 4,
                        resource: {
                            buffer: this.countsBuffer!,
                            size: countsSize,
                        },
                    },
                ],
            });

            const numWorkgroupsScatter = Math.ceil(
                numWorkgroups / this.workgroupSize,
            );

            passEncoder.setPipeline(this.scatterPipelines![iter]);
            passEncoder.setBindGroup(0, scatterBindGroup);
            passEncoder.dispatchWorkgroups(numWorkgroupsScatter);
        }
    }
}

export default WebGPUSort;
