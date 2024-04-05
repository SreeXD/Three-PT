import WebGPUContext from "./WebGPUContext";
import WebGPUSort from "./WebGPUSort";
import WebGPUResourceGroup from "./WebGPUResourceGroup";
import {
    computeRootBoundsShader,
    computeParentBoundsShader,
    computeMortonCodesShader,
    computeBVHShader,
    computeBVHBoundsShader,
} from "./shaders/bvh";

import {
    nodesToLinesShader,
    drawBVHVertexShader,
    drawBVHFragmentShader,
} from "./shaders/debugBVH";

class WebGPUBVH {
    context: WebGPUContext;
    debug: boolean;
    computeBoundsWorkgroupSize: number;
    computeBoundsReduceSize: number;
    computeMortonCodesWorkgroupSize: number;
    computeBVHWorkgroupSize: number;
    computeBVHBoundsWorkgroupSize: number;
    nodesToLinesWorkgroupSize: number;
    gpuSort: WebGPUSort;

    computeRootBoundsPipeline?: GPUComputePipeline;
    computeParentBoundsPipeline?: GPUComputePipeline;
    computeMortonCodesPipeline?: GPUComputePipeline;
    computeBVHPipeline?: GPUComputePipeline;
    computeBVHBoundsPipeline?: GPUComputePipeline;
    nodesToLinesPipeline?: GPUComputePipeline;
    drawBVHPipeline?: GPURenderPipeline;

    outputAABBsBuffers?: GPUBuffer[];
    aabbBuffer?: GPUBuffer;
    idsBuffer?: GPUBuffer;
    mortonCodesBuffer?: GPUBuffer;
    nodeAtomicsBuffer?: GPUBuffer;
    linesVertexBuffer?: GPUBuffer;
    linesLevelsBuffer?: GPUBuffer;

    constructor(
        context: WebGPUContext,
        debug: boolean = false,
        computeBoundsWorkgroupSize = 32,
        computeBoundsReduceSize = 128,
        computeMortonCodesWorkgroupSize = 128,
        computeBVHWorkgroupSize = 64,
        computeBVHBoundsWorkgroupSize = 64,
        nodesToLinesWorkgroupSize = 64,
    ) {
        this.context = context;
        this.debug = debug;
        this.computeBoundsWorkgroupSize = computeBoundsWorkgroupSize;
        this.computeBoundsReduceSize = computeBoundsReduceSize;
        this.computeMortonCodesWorkgroupSize = computeMortonCodesWorkgroupSize;
        this.computeBVHWorkgroupSize = computeBVHWorkgroupSize;
        this.computeBVHBoundsWorkgroupSize = computeBVHBoundsWorkgroupSize;
        this.nodesToLinesWorkgroupSize = nodesToLinesWorkgroupSize;
        this.gpuSort = new WebGPUSort(this.context);
    }

    async initialize() {
        if (!this.context.device) {
            await this.context.initialize();
        }

        const computeRootBoundsModule = this.context.device!.createShaderModule(
            {
                code: computeRootBoundsShader(
                    this.computeBoundsWorkgroupSize,
                    this.computeBoundsReduceSize,
                ),
            },
        );

        const computeParentBoundsModule =
            this.context.device!.createShaderModule({
                code: computeParentBoundsShader(
                    this.computeBoundsWorkgroupSize,
                    this.computeBoundsReduceSize,
                ),
            });

        const computeBoundsBindGroupLayout =
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

        this.computeRootBoundsPipeline =
            this.context.device!.createComputePipeline({
                layout: this.context.device!.createPipelineLayout({
                    bindGroupLayouts: [computeBoundsBindGroupLayout],
                }),
                compute: {
                    module: computeRootBoundsModule,
                    entryPoint: "main",
                },
            });

        this.computeParentBoundsPipeline =
            this.context.device!.createComputePipeline({
                layout: this.context.device!.createPipelineLayout({
                    bindGroupLayouts: [computeBoundsBindGroupLayout],
                }),
                compute: {
                    module: computeParentBoundsModule,
                    entryPoint: "main",
                },
            });

        const computeMortonCodesModule =
            this.context.device!.createShaderModule({
                code: computeMortonCodesShader(
                    this.computeMortonCodesWorkgroupSize,
                ),
            });

        const computeMortonCodesBindGroupLayout =
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
                            type: "storage",
                        },
                    },
                ],
            });

        this.computeMortonCodesPipeline =
            this.context.device!.createComputePipeline({
                layout: this.context.device!.createPipelineLayout({
                    bindGroupLayouts: [computeMortonCodesBindGroupLayout],
                }),
                compute: {
                    module: computeMortonCodesModule,
                    entryPoint: "main",
                },
            });

        const computeBVHModule = this.context.device!.createShaderModule({
            code: computeBVHShader(this.computeBVHWorkgroupSize),
        });

        const computeBVHBindGroupLayout =
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
                ],
            });

        this.computeBVHPipeline = this.context.device!.createComputePipeline({
            layout: this.context.device!.createPipelineLayout({
                bindGroupLayouts: [computeBVHBindGroupLayout],
            }),
            compute: {
                module: computeBVHModule,
                entryPoint: "main",
            },
        });

        const computeBVHBoundsModule = this.context.device!.createShaderModule({
            code: computeBVHBoundsShader(this.computeBVHBoundsWorkgroupSize),
        });

        const computeBVHBoundsBindGroupLayout =
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

        this.computeBVHBoundsPipeline =
            this.context.device!.createComputePipeline({
                layout: this.context.device!.createPipelineLayout({
                    bindGroupLayouts: [computeBVHBoundsBindGroupLayout],
                }),
                compute: {
                    module: computeBVHBoundsModule,
                    entryPoint: "main",
                },
            });

        await this.gpuSort.initialize();

        if (this.debug) {
            const nodesToLinesModule = this.context.device!.createShaderModule({
                code: nodesToLinesShader(this.nodesToLinesWorkgroupSize),
            });

            const nodesToLinesBindGroupLayout =
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
                    ],
                });

            this.nodesToLinesPipeline =
                this.context.device!.createComputePipeline({
                    layout: this.context.device!.createPipelineLayout({
                        bindGroupLayouts: [nodesToLinesBindGroupLayout],
                    }),
                    compute: {
                        module: nodesToLinesModule,
                        entryPoint: "main",
                    },
                });

            const drawBVHVertexModule = this.context.device!.createShaderModule(
                {
                    code: drawBVHVertexShader,
                },
            );

            const drawBVHFragmentModule =
                this.context.device!.createShaderModule({
                    code: drawBVHFragmentShader,
                });

            const drawBVHBindGroupLayout =
                this.context.device!.createBindGroupLayout({
                    entries: [
                        {
                            binding: 0,
                            visibility: GPUShaderStage.VERTEX,
                            buffer: {
                                type: "uniform",
                            },
                        },
                        {
                            binding: 1,
                            visibility: GPUShaderStage.VERTEX,
                            buffer: {
                                type: "uniform",
                            },
                        },
                        {
                            binding: 2,
                            visibility: GPUShaderStage.FRAGMENT,
                            sampler: {
                                type: "non-filtering",
                            },
                        },
                        {
                            binding: 3,
                            visibility: GPUShaderStage.FRAGMENT,
                            texture: {
                                sampleType: "unfilterable-float",
                                viewDimension: "2d",
                            },
                        },
                        {
                            binding: 4,
                            visibility: GPUShaderStage.FRAGMENT,
                            buffer: {
                                type: "uniform",
                            },
                        },
                    ],
                });

            this.drawBVHPipeline = this.context.device!.createRenderPipeline({
                layout: this.context.device!.createPipelineLayout({
                    bindGroupLayouts: [drawBVHBindGroupLayout],
                }),
                vertex: {
                    module: drawBVHVertexModule,
                    entryPoint: "main",
                    buffers: [
                        {
                            arrayStride: 16,
                            attributes: [
                                {
                                    shaderLocation: 0,
                                    offset: 0,
                                    format: "float32x4",
                                },
                            ],
                        },
                        {
                            arrayStride: 4,
                            attributes: [
                                {
                                    shaderLocation: 1,
                                    offset: 0,
                                    format: "uint32",
                                },
                            ],
                        },
                    ],
                },
                fragment: {
                    module: drawBVHFragmentModule,
                    entryPoint: "main",

                    // @ts-ignore
                    targets: [
                        {
                            format: "rgba8unorm",
                            blend: {
                                alpha: {
                                    dstFactor: "one-minus-src-alpha",
                                    srcFactor: "src-alpha",
                                    operation: "add",
                                },
                                color: {
                                    dstFactor: "one-minus-src-alpha",
                                    srcFactor: "src-alpha",
                                    operation: "add",
                                },
                            },
                        },
                    ],
                },
                primitive: {
                    topology: "line-list",
                },
            });
        }
    }

    allocate(
        resourceGroup: WebGPUResourceGroup,
        maxPositionsLength: number,
        maxTrianglesLength: number,
    ) {
        this.outputAABBsBuffers = [];
        let currentLength = maxPositionsLength;

        do {
            currentLength = Math.ceil(
                currentLength / this.computeBoundsReduceSize,
            );

            this.outputAABBsBuffers.push(
                resourceGroup.createBuffer({
                    size: currentLength * 32,
                    usage: GPUBufferUsage.STORAGE,
                }),
            );
        } while (currentLength > 1);

        this.aabbBuffer = resourceGroup.createBuffer({
            size: 32,
            usage: GPUBufferUsage.STORAGE,
        });

        this.mortonCodesBuffer = resourceGroup.createBuffer({
            size: maxTrianglesLength * 4,
            usage: GPUBufferUsage.STORAGE,
        });

        this.idsBuffer = resourceGroup.createBuffer({
            size: maxTrianglesLength * 4,
            usage: GPUBufferUsage.STORAGE,
        });

        this.nodeAtomicsBuffer = resourceGroup.createBuffer({
            size: (maxTrianglesLength - 1) * 4,
            usage: GPUBufferUsage.STORAGE,
        });

        this.gpuSort.allocate(resourceGroup, maxTrianglesLength);

        if (this.debug) {
            this.linesVertexBuffer = resourceGroup.createBuffer({
                size: (maxTrianglesLength - 1) * 24 * 16,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX,
            });

            this.linesLevelsBuffer = resourceGroup.createBuffer({
                size: (maxTrianglesLength - 1) * 24 * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX,
            });
        }
    }

    private encodeComputeBoundsPass(
        passEncoder: GPUComputePassEncoder,
        inputBuffer: GPUBuffer,
        inputSize: number = inputBuffer.size,
        inputsAreAABBs: boolean = false,
        depth: number = 0,
    ) {
        if (!this.context.device) {
            throw new Error("WebGPUBVH: WebGPUContext is not initialized");
        }

        const inputLength = inputSize / (inputsAreAABBs ? 32 : 4);
        const numReduced = Math.ceil(
            inputLength / this.computeBoundsReduceSize,
        );
        const numWorkgroups = Math.ceil(
            numReduced / this.computeBoundsWorkgroupSize,
        );
        const recurseNeeded = numReduced > 1;

        if (recurseNeeded) {
            var outputAABBsBufferSize = numReduced * 32;
        }

        const computeBoundsBindGroup = this.context.device.createBindGroup({
            layout: this.computeRootBoundsPipeline!.getBindGroupLayout(0),
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
                    resource: !recurseNeeded
                        ? {
                              buffer: this.aabbBuffer!,
                          }
                        : {
                              buffer: this.outputAABBsBuffers![depth],
                              size: outputAABBsBufferSize!,
                          },
                },
            ],
        });

        passEncoder.setPipeline(
            !inputsAreAABBs
                ? this.computeRootBoundsPipeline!
                : this.computeParentBoundsPipeline!,
        );
        passEncoder.setBindGroup(0, computeBoundsBindGroup);
        passEncoder.dispatchWorkgroups(numWorkgroups);

        if (recurseNeeded) {
            this.encodeComputeBoundsPass(
                passEncoder,
                this.outputAABBsBuffers![depth],
                outputAABBsBufferSize!,
                true,
                depth + 1,
            );
        }
    }

    private encodeComputeMortonCodesPass(
        passEncoder: GPUComputePassEncoder,
        positionsBuffer: GPUBuffer,
        positionsSize: number,
        indexBuffer: GPUBuffer,
        indexSize: number,
    ) {
        if (!this.context.device) {
            throw new Error("WebGPUBVH: WebGPUContext is not initialized");
        }

        const indexLength = indexBuffer.size / 4;
        const trianglesLength = indexLength / 3;

        const numWorkgroups = Math.ceil(
            trianglesLength / this.computeMortonCodesWorkgroupSize,
        );

        const computeMortonCodesBindGroup = this.context.device.createBindGroup(
            {
                layout: this.computeMortonCodesPipeline!.getBindGroupLayout(0),
                entries: [
                    {
                        binding: 0,
                        resource: {
                            buffer: positionsBuffer,
                            size: positionsSize,
                        },
                    },
                    {
                        binding: 1,
                        resource: {
                            buffer: indexBuffer,
                            size: indexSize,
                        },
                    },
                    {
                        binding: 2,
                        resource: {
                            buffer: this.aabbBuffer!,
                        },
                    },
                    {
                        binding: 3,
                        resource: {
                            buffer: this.mortonCodesBuffer!,
                            size: trianglesLength * 4,
                        },
                    },
                ],
            },
        );

        passEncoder.setPipeline(this.computeMortonCodesPipeline!);
        passEncoder.setBindGroup(0, computeMortonCodesBindGroup);
        passEncoder.dispatchWorkgroups(numWorkgroups);
    }

    encodeComputeBVHPass(
        passEncoder: GPUComputePassEncoder,
        positionsBuffer: GPUBuffer,
        positionsSize: number,
        indexBuffer: GPUBuffer,
        indexSize: number,
        nodesBuffer: GPUBuffer,
        nodeBoundsBuffer: GPUBuffer,
    ) {
        if (!this.context.device) {
            throw new Error("WebGPUBVH: WebGPUContext is not initialized");
        }

        this.encodeComputeBoundsPass(passEncoder, positionsBuffer);

        const trianglesLength = indexBuffer.size / 12;
        const mortonCodesSize = trianglesLength * 4;

        this.encodeComputeMortonCodesPass(
            passEncoder,
            positionsBuffer,
            positionsSize,
            indexBuffer,
            indexSize,
        );

        this.gpuSort.encodeSortPass(
            passEncoder,
            this.mortonCodesBuffer!,
            mortonCodesSize,
            this.idsBuffer!,
        );

        const computeBVHBindGroup = this.context.device.createBindGroup({
            layout: this.computeBVHPipeline!.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: this.mortonCodesBuffer!,
                        size: mortonCodesSize,
                    },
                },
                {
                    binding: 1,
                    resource: {
                        buffer: this.idsBuffer!,
                        size: mortonCodesSize,
                    },
                },
                {
                    binding: 2,
                    resource: {
                        buffer: nodesBuffer,
                        size: (trianglesLength - 1) * 20,
                    },
                },
            ],
        });

        const computeBVHNumWorkgroups = Math.ceil(
            (trianglesLength - 1) / this.computeBVHWorkgroupSize,
        );

        passEncoder.setPipeline(this.computeBVHPipeline!);
        passEncoder.setBindGroup(0, computeBVHBindGroup);
        passEncoder.dispatchWorkgroups(computeBVHNumWorkgroups);

        const computeBVHBoundsBindGroup = this.context.device.createBindGroup({
            layout: this.computeBVHBoundsPipeline!.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: positionsBuffer,
                        size: positionsSize,
                    },
                },
                {
                    binding: 1,
                    resource: {
                        buffer: indexBuffer,
                        size: indexSize,
                    },
                },
                {
                    binding: 2,
                    resource: {
                        buffer: nodesBuffer,
                        size: (trianglesLength - 1) * 20,
                    },
                },
                {
                    binding: 3,
                    resource: {
                        buffer: nodeBoundsBuffer,
                        size: (trianglesLength - 1) * 32,
                    },
                },
                {
                    binding: 4,
                    resource: {
                        buffer: this.nodeAtomicsBuffer!,
                        size: (trianglesLength - 1) * 4,
                    },
                },
            ],
        });

        const computeBVHBoundsNumWorkgroups = Math.ceil(
            (trianglesLength - 1) / this.computeBVHBoundsWorkgroupSize,
        );

        passEncoder.setPipeline(this.computeBVHBoundsPipeline!);
        passEncoder.setBindGroup(0, computeBVHBoundsBindGroup);
        passEncoder.dispatchWorkgroups(computeBVHBoundsNumWorkgroups);
    }

    encodeNodesToLinesPass(
        passEncoder: GPUComputePassEncoder,
        nodesBuffer: GPUBuffer,
        nodeBoundsBuffer: GPUBuffer,
    ) {
        if (!this.context.device) {
            throw new Error("WebGPUBVH: WebGPUContext is not initialized");
        }

        if (!this.debug) {
            throw new Error("WebGPUBVH: debug flag was not set");
        }

        const nodesLength = nodesBuffer.size / 20;

        const nodesToLinesBindGroup = this.context.device.createBindGroup({
            layout: this.nodesToLinesPipeline!.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: nodesBuffer,
                        size: nodesBuffer.size,
                    },
                },
                {
                    binding: 1,
                    resource: {
                        buffer: nodeBoundsBuffer,
                        size: nodeBoundsBuffer.size,
                    },
                },
                {
                    binding: 2,
                    resource: {
                        buffer: this.linesVertexBuffer!,
                        size: nodesLength * 24 * 16,
                    },
                },
                {
                    binding: 3,
                    resource: {
                        buffer: this.linesLevelsBuffer!,
                        size: nodesLength * 24 * 4,
                    },
                },
            ],
        });

        const numWorkgroups = Math.ceil(
            nodesLength / this.nodesToLinesWorkgroupSize,
        );

        passEncoder.setPipeline(this.nodesToLinesPipeline!);
        passEncoder.setBindGroup(0, nodesToLinesBindGroup);
        passEncoder.dispatchWorkgroups(numWorkgroups);
    }

    encodeDrawBVHPass(
        passEncoder: GPURenderPassEncoder,
        nodesLength: number,
        viewMatrixBuffer: GPUBuffer,
        projectionMatrixBuffer: GPUBuffer,
        depthTextureSampler: GPUSampler,
        depthTextureView: GPUTextureView,
        drawBVHOptionsBuffer: GPUBuffer,
    ) {
        if (!this.context.device) {
            throw new Error("WebGPUBVH: WebGPUContext is not initialized");
        }

        if (!this.debug) {
            throw new Error("WebGPUBVH: debug flag was not set");
        }

        const drawBVHBindGroup = this.context.device!.createBindGroup({
            layout: this.drawBVHPipeline!.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: viewMatrixBuffer,
                    },
                },
                {
                    binding: 1,
                    resource: {
                        buffer: projectionMatrixBuffer,
                    },
                },
                {
                    binding: 2,
                    resource: depthTextureSampler,
                },
                {
                    binding: 3,
                    resource: depthTextureView,
                },
                {
                    binding: 4,
                    resource: {
                        buffer: drawBVHOptionsBuffer,
                    },
                },
            ],
        });

        passEncoder.setPipeline(this.drawBVHPipeline!);
        passEncoder.setVertexBuffer(0, this.linesVertexBuffer!);
        passEncoder.setVertexBuffer(1, this.linesLevelsBuffer!);
        passEncoder.setBindGroup(0, drawBVHBindGroup);
        passEncoder.draw(nodesLength * 24);
    }
}

export default WebGPUBVH;
