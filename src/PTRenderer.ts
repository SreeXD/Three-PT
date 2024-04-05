import {
    Renderer,
    Object3D,
    Mesh,
    BufferGeometry,
    NormalBufferAttributes,
    PerspectiveCamera,
    MeshStandardMaterial,
} from "three";

import {
    WebGPUContext,
    WebGPUContextOptions,
    WebGPUPathTrace,
    WebGPUResourceGroup,
    WebGPUBVH,
} from "./core";

export type PTRendererOptions = {
    canvas: HTMLCanvasElement;
    numBounces: number;
    debug?: {
        logs?: boolean;
        drawBVH?: boolean;
    };
};

export type PTRendererLogInfo = {
    bvhConstructionTime: number;
    pathTraceTime: number;
    maxIntersectionChecks: number;
    averageIntersectionChecks: number;
    totalIntersectionChecks: number;
    maxBVHTraversalStackSize: number;
};

class PTRenderer implements Renderer {
    domElement: HTMLCanvasElement;
    options: PTRendererOptions;
    private webGPUContext: WebGPUContext;
    private webGPUPathTrace: WebGPUPathTrace;
    private canvasContext?: GPUCanvasContext;
    private webGPUResourceGroup?: WebGPUResourceGroup;

    scene?: Object3D;
    camera?: PerspectiveCamera;
    pathTraceData: Uint32Array;
    private sceneObjectsBuffer?: GPUBuffer;
    private sceneLightsBuffer?: GPUBuffer;
    private scenePositionsBuffer?: GPUBuffer;
    private sceneNormalsBuffer?: GPUBuffer;
    private sceneIndexBuffer?: GPUBuffer;

    // map each triangle to the object it belongs to
    private sceneTriangleToObjectBuffer?: GPUBuffer;
    private cameraArray?: Float32Array;
    private cameraBuffer?: GPUBuffer;
    private pathTraceDataBuffer?: GPUBuffer;
    private cumulativeBuffer?: GPUBuffer;

    private webGPUBVH?: WebGPUBVH;
    private sceneBVHNodesBuffer?: GPUBuffer;
    private sceneBVHNodeBoundsBuffer?: GPUBuffer;

    private querySet?: GPUQuerySet;
    private debugBuffer?: GPUBuffer;
    private debugReadBuffer?: GPUBuffer;
    private drawBVHViewMatrixBuffer?: GPUBuffer;
    private drawBVHProjectionMatrixBuffer?: GPUBuffer;
    private drawBVHOptions?: DataView;
    private drawBVHOptionsBuffer?: GPUBuffer;
    private depthTextureSampler?: GPUSampler;
    private depthTexture?: GPUTexture;

    constructor(options: PTRendererOptions) {
        this.options = options;
        this.domElement = this.options.canvas;
        this.pathTraceData = new Uint32Array([this.options.numBounces, 0]);

        const webGPUContextOptions: WebGPUContextOptions = {};

        if (this.options.debug?.logs) {
            webGPUContextOptions.requiredFeatures = ["timestamp-query"];
        }

        this.webGPUContext = new WebGPUContext(webGPUContextOptions);

        this.webGPUPathTrace = new WebGPUPathTrace(
            this.webGPUContext,
            this.options.numBounces,
            this.options.debug?.logs,
            this.options.debug?.drawBVH,
        );

        this.webGPUBVH = new WebGPUBVH(
            this.webGPUContext,
            this.options.debug?.drawBVH,
        );
    }

    async initialize() {
        await this.webGPUContext.initialize();
        await this.webGPUPathTrace.initialize();
        await this.webGPUBVH!.initialize();

        this.webGPUResourceGroup = this.webGPUContext.createResourceGroup();

        const canvasContext = this.domElement.getContext("webgpu");

        if (!canvasContext) {
            throw new Error(
                "PTRenderer: couldn't create webgpu canvas context",
            );
        }

        canvasContext.configure({
            device: this.webGPUContext.device!,
            format: "rgba8unorm",
            alphaMode: "premultiplied",
            usage:
                GPUTextureUsage.RENDER_ATTACHMENT |
                GPUTextureUsage.STORAGE_BINDING |
                GPUTextureUsage.COPY_SRC,
        });

        this.canvasContext = canvasContext;

        this.pathTraceDataBuffer = this.webGPUResourceGroup!.createBuffer({
            size: 32,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        const canvasTexture = canvasContext.getCurrentTexture();

        this.cumulativeBuffer = this.webGPUResourceGroup!.createBuffer({
            size: canvasTexture.width * canvasTexture.height * 16,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        if (this.options.debug?.logs) {
            /*
                format:
                    0: bvh construction pass begin timestamp
                    1: bvh construction pass end timestamp
                    2: path trace pass begin timestamp
                    3. path trace pass end timestamp 
                    4: maximum intersection checks per invocation
                    5: total intersection checks
                    6: maximum bvh traversal stack size
            */
            this.querySet = this.webGPUResourceGroup.createQuerySet({
                count: 4,
                type: "timestamp",
            });

            this.debugBuffer = this.webGPUResourceGroup.createBuffer({
                usage:
                    GPUBufferUsage.STORAGE |
                    GPUBufferUsage.QUERY_RESOLVE |
                    GPUBufferUsage.COPY_SRC |
                    GPUBufferUsage.COPY_DST,
                size: 44,
            });

            this.debugReadBuffer = this.webGPUResourceGroup.createBuffer({
                usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
                size: 44,
            });
        }

        if (this.options.debug?.drawBVH) {
            /*
                format:
                    0: line color
                    1: level to draw
                    2: draw parents flag
            */
            this.drawBVHOptionsBuffer = this.webGPUResourceGroup!.createBuffer({
                size: 32,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            });

            this.drawBVHOptions = new DataView(new ArrayBuffer(24));
            this.drawBVHOptions.setFloat32(0, 0.25, true);
            this.drawBVHOptions.setFloat32(4, 0.9, true);
            this.drawBVHOptions.setFloat32(8, 0.25, true);
            this.drawBVHOptions.setFloat32(12, 0.3, true);
            this.drawBVHOptions.setUint32(16, 5, true);
            this.drawBVHOptions.setUint32(20, 1, true);

            this.webGPUContext.device!.queue.writeBuffer(
                this.drawBVHOptionsBuffer,
                0,
                this.drawBVHOptions,
                0,
                24,
            );

            this.depthTextureSampler =
                this.webGPUContext.device!.createSampler();

            this.depthTexture = this.webGPUResourceGroup!.createTexture({
                format: "r32float",
                usage:
                    GPUTextureUsage.STORAGE_BINDING |
                    GPUTextureUsage.TEXTURE_BINDING,
                size: {
                    width: this.domElement.width,
                    height: this.domElement.height,
                },
            });
        }
    }

    setSize(width: number, height: number) {
        this.domElement.width = width;
        this.domElement.height = height;
    }

    private updateScene() {
        const sceneObjects = [];
        const sceneLights = [];
        const scenePositions = [];
        const sceneNormals = [];
        const sceneIndex = [];
        const sceneTriangleToObject = [];
        let sceneObjectId = 0;

        const stack: Object3D[] = [];
        stack.push(this.scene!);

        while (stack.length) {
            const object = stack.pop()!;

            if (object instanceof Mesh) {
                const mesh = object as Mesh<
                    BufferGeometry<NormalBufferAttributes>,
                    MeshStandardMaterial
                >;

                const positions = mesh.geometry.getAttribute("position").array;
                const normals = mesh.geometry.getAttribute("normal").array;
                const index = mesh.geometry.getIndex()!.array;
                const offset = scenePositions.length / 3;

                sceneObjects.push(
                    sceneIndex.length,
                    index.length,
                    0,
                    0,
                    mesh.material.color.r,
                    mesh.material.color.g,
                    mesh.material.color.b,
                    1,
                    mesh.material.emissive.r,
                    mesh.material.emissive.g,
                    mesh.material.emissive.b,
                    mesh.material.emissiveIntensity,
                    mesh.material.roughness,
                    mesh.material.metalness,
                    0,
                    0,
                );

                if (
                    mesh.material.emissive.getHex() > 0 &&
                    mesh.material.emissiveIntensity > 0
                ) {
                    sceneLights.push(sceneObjectId);
                }

                for (let i = 0; i < positions.length; i++) {
                    scenePositions.push(positions[i]);
                    sceneNormals.push(normals[i]);
                }

                for (let i = 0; i < index.length; i++) {
                    sceneIndex.push(offset + index[i]);
                }

                for (let i = 0; i < index.length / 3; i++) {
                    sceneTriangleToObject.push(sceneObjectId);
                }

                sceneObjectId++;
            }

            for (const child of object.children) {
                stack.push(child);
            }
        }

        this.sceneObjectsBuffer = this.webGPUResourceGroup!.createBuffer({
            size: sceneObjects.length * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        this.sceneLightsBuffer = this.webGPUResourceGroup!.createBuffer({
            size: sceneObjects.length * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        this.scenePositionsBuffer = this.webGPUResourceGroup!.createBuffer({
            size: scenePositions.length * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        this.sceneNormalsBuffer = this.webGPUResourceGroup!.createBuffer({
            size: sceneNormals.length * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        this.sceneIndexBuffer = this.webGPUResourceGroup!.createBuffer({
            size: sceneIndex.length * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        this.sceneTriangleToObjectBuffer =
            this.webGPUResourceGroup!.createBuffer({
                size: sceneTriangleToObject.length * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            });

        const sceneObjectsDataView = new DataView(
            new ArrayBuffer(sceneObjects.length * 4),
        );

        for (let i = 0; i < sceneObjects.length; i += 16) {
            for (let j = 0; j < 2; j++) {
                sceneObjectsDataView.setUint32(
                    4 * (i + j),
                    sceneObjects[i + j],
                    true,
                );
            }

            for (let j = 2; j < 14; j++) {
                sceneObjectsDataView.setFloat32(
                    4 * (i + j),
                    sceneObjects[i + j],
                    true,
                );
            }
        }

        this.webGPUContext.device!.queue.writeBuffer(
            this.sceneObjectsBuffer,
            0,
            sceneObjectsDataView,
        );

        this.webGPUContext.device!.queue.writeBuffer(
            this.sceneLightsBuffer,
            0,
            new Float32Array(sceneLights),
        );

        this.webGPUContext.device!.queue.writeBuffer(
            this.scenePositionsBuffer,
            0,
            new Float32Array(scenePositions),
        );

        this.webGPUContext.device!.queue.writeBuffer(
            this.sceneNormalsBuffer,
            0,
            new Float32Array(sceneNormals),
        );

        this.webGPUContext.device!.queue.writeBuffer(
            this.sceneIndexBuffer,
            0,
            new Uint32Array(sceneIndex),
        );

        this.webGPUContext.device!.queue.writeBuffer(
            this.sceneTriangleToObjectBuffer,
            0,
            new Uint32Array(sceneTriangleToObject),
        );

        const trianglesLength = sceneIndex.length / 3;

        this.webGPUBVH!.allocate(
            this.webGPUResourceGroup!,
            scenePositions.length,
            trianglesLength,
        );

        this.sceneBVHNodesBuffer = this.webGPUResourceGroup!.createBuffer({
            size: (trianglesLength - 1) * 20,
            usage: GPUBufferUsage.STORAGE,
        });

        this.sceneBVHNodeBoundsBuffer = this.webGPUResourceGroup!.createBuffer({
            size: (trianglesLength - 1) * 32,
            usage: GPUBufferUsage.STORAGE,
        });

        const commandEncoder =
            this.webGPUContext.device!.createCommandEncoder();

        if (this.options.debug?.logs) {
            // @ts-ignore
            commandEncoder.writeTimestamp(this.querySet!, 0);
        }

        const passEncoder = commandEncoder.beginComputePass();

        this.webGPUBVH!.encodeComputeBVHPass(
            passEncoder,
            this.scenePositionsBuffer,
            this.scenePositionsBuffer.size,
            this.sceneIndexBuffer,
            this.sceneIndexBuffer.size,
            this.sceneBVHNodesBuffer,
            this.sceneBVHNodeBoundsBuffer,
        );

        passEncoder.end();

        if (this.options.debug?.logs) {
            // @ts-ignore
            commandEncoder.writeTimestamp(this.querySet!, 1);
        }

        const gpuCommands = commandEncoder.finish();
        this.webGPUContext.device!.queue.submit([gpuCommands]);

        if (this.options.debug?.drawBVH) {
            this.drawBVHViewMatrixBuffer =
                this.webGPUResourceGroup!.createBuffer({
                    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
                    size: 64,
                });

            this.drawBVHProjectionMatrixBuffer =
                this.webGPUResourceGroup!.createBuffer({
                    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
                    size: 64,
                });

            const drawBVHCommandEncoder =
                this.webGPUContext.device!.createCommandEncoder();

            const drawBVHPassEncoder = drawBVHCommandEncoder.beginComputePass();

            this.webGPUBVH!.encodeNodesToLinesPass(
                drawBVHPassEncoder,
                this.sceneBVHNodesBuffer!,
                this.sceneBVHNodeBoundsBuffer!,
            );

            drawBVHPassEncoder.end();

            const drawBVHGPUCommands = drawBVHCommandEncoder.finish();
            this.webGPUContext.device!.queue.submit([drawBVHGPUCommands]);
        }
    }

    private updateCamera() {
        this.camera!.updateMatrixWorld(true);

        const position = this.camera!.position;
        const viewMatrix = this.camera!.matrixWorld.elements;
        const near = this.camera!.near;
        const far = this.camera!.far;
        const fov = this.camera!.fov;

        if (!this.cameraArray) {
            this.cameraArray = new Float32Array(16);
        }

        const cameraArrayValues = new Float32Array([
            position.x,
            position.y,
            position.z,
            near,
            viewMatrix[0],
            viewMatrix[1],
            viewMatrix[2],
            far,
            viewMatrix[4],
            viewMatrix[5],
            viewMatrix[6],
            fov,
            -viewMatrix[8],
            -viewMatrix[9],
            -viewMatrix[10],
            1,
        ]);

        const clear = cameraArrayValues.reduce(
            (f, x, i) => f || x != this.cameraArray![i],
            false,
        );

        this.cameraArray.set(cameraArrayValues);

        if (!this.cameraBuffer) {
            this.cameraBuffer = this.webGPUResourceGroup!.createBuffer({
                size: 64,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            });
        }

        this.webGPUContext.device!.queue.writeBuffer(
            this.cameraBuffer,
            0,
            this.cameraArray,
        );

        return clear;
    }

    private update(scene: Object3D, camera: PerspectiveCamera): boolean {
        if (!this.webGPUContext.device) {
            throw new Error("PTRenderer: not initialized");
        }

        if (!this.sceneObjectsBuffer) {
            this.scene = scene;
            this.updateScene();
        }

        this.camera = camera;
        return this.updateCamera();
    }

    render(scene: Object3D, camera: PerspectiveCamera) {
        if (!this.webGPUContext.device) {
            throw new Error("PTRenderer: not initialized");
        }

        const clear = this.update(scene, camera);

        this.pathTraceData[0] = clear ? 1 : this.pathTraceData[0] + 1;

        this.webGPUContext.device!.queue.writeBuffer(
            this.pathTraceDataBuffer!,
            0,
            this.pathTraceData,
            0,
            1,
        );

        const commandEncoder = this.webGPUContext.device.createCommandEncoder();

        if (clear) {
            commandEncoder.clearBuffer(this.cumulativeBuffer!);
        }

        const canvasTexture = this.canvasContext!.getCurrentTexture();

        if (this.options.debug?.logs) {
            commandEncoder.clearBuffer(this.debugBuffer!, 32);

            // @ts-ignore
            commandEncoder.writeTimestamp(this.querySet!, 2);
        }

        const passEncoder = commandEncoder.beginComputePass();

        this.webGPUPathTrace.encodePathTracePass(
            passEncoder,
            this.sceneObjectsBuffer!,
            this.sceneLightsBuffer!,
            this.scenePositionsBuffer!,
            this.sceneNormalsBuffer!,
            this.sceneIndexBuffer!,
            this.sceneTriangleToObjectBuffer!,
            this.cameraBuffer!,
            this.pathTraceDataBuffer!,
            this.cumulativeBuffer!,
            canvasTexture.createView(),
            this.domElement.width,
            this.domElement.height,
            this.sceneBVHNodesBuffer!,
            this.sceneBVHNodeBoundsBuffer!,
            this.debugBuffer,
            this.depthTexture?.createView(),
        );

        passEncoder.end();

        if (this.options.debug?.logs) {
            // @ts-ignore
            commandEncoder.writeTimestamp(this.querySet!, 3);

            commandEncoder.resolveQuerySet(
                this.querySet!,
                0,
                4,
                this.debugBuffer!,
                0,
            );
        }

        const gpuCommands = commandEncoder.finish();
        this.webGPUContext.device.queue.submit([gpuCommands]);
    }

    async debugLogInfo(): Promise<PTRendererLogInfo> {
        if (!this.webGPUContext.device) {
            throw new Error("PTRenderer: not initialized");
        }

        if (!this.options.debug?.logs) {
            throw new Error("PTRenderer: debug.logs flag is not set");
        }

        const debugArrayBytes = await this.webGPUResourceGroup!.readBuffer(
            this.debugBuffer!,
            this.debugReadBuffer!,
        );

        const debugArrayValues = new DataView(debugArrayBytes);

        // @ts-ignore
        const bvhConstructionBeginTimestamp = debugArrayValues.getBigUint64(
            0,
            true,
        );
        // @ts-ignore
        const bvhConstructionEndTimestamp = debugArrayValues.getBigUint64(
            8,
            true,
        );
        const bvhConstructionTime =
            Number(
                bvhConstructionEndTimestamp - bvhConstructionBeginTimestamp,
            ) / 1e6;

        // @ts-ignore
        const pathTraceBeginTimestamp = debugArrayValues.getBigUint64(16, true);
        // @ts-ignore
        const pathTraceEndTimestamp = debugArrayValues.getBigUint64(24, true);
        const pathTraceTime =
            Number(pathTraceEndTimestamp - pathTraceBeginTimestamp) / 1e6;

        const maxIntersectionChecks = debugArrayValues.getUint32(32, true);
        const totalIntersectionChecks = debugArrayValues.getUint32(36, true);

        const totalInvocations = this.domElement.width * this.domElement.height;

        const averageIntersectionChecks = Math.ceil(
            totalIntersectionChecks / totalInvocations,
        );

        const maxBVHTraversalStackSize = debugArrayValues.getUint32(40, true);

        return {
            bvhConstructionTime,
            pathTraceTime,
            maxIntersectionChecks,
            averageIntersectionChecks,
            totalIntersectionChecks,
            maxBVHTraversalStackSize,
        };
    }

    debugSetDrawBVHColor(r: number, g: number, b: number, a: number) {
        if (!this.webGPUContext.device) {
            throw new Error("PTRenderer: not initialized");
        }

        if (!this.options.debug?.drawBVH) {
            throw new Error("PTRenderer: debug.drawBVH flag is not set");
        }

        this.drawBVHOptions!.setFloat32(0, r, true);
        this.drawBVHOptions!.setFloat32(4, g, true);
        this.drawBVHOptions!.setFloat32(8, b, true);
        this.drawBVHOptions!.setFloat32(12, a, true);

        this.webGPUContext.device.queue.writeBuffer(
            this.drawBVHOptionsBuffer!,
            0,
            this.drawBVHOptions!,
        );
    }

    debugSetDrawBVHLevel(level: number) {
        if (!this.webGPUContext.device) {
            throw new Error("PTRenderer: not initialized");
        }

        if (!this.options.debug?.drawBVH) {
            throw new Error("PTRenderer: debug.drawBVH flag is not set");
        }

        this.drawBVHOptions!.setUint32(16, level, true);

        this.webGPUContext.device.queue.writeBuffer(
            this.drawBVHOptionsBuffer!,
            16,
            this.drawBVHOptions!,
            16,
            4,
        );
    }

    debugSetDrawBVHShowParents(showParents: boolean) {
        if (!this.webGPUContext.device) {
            throw new Error("PTRenderer: not initialized");
        }

        if (!this.options.debug?.drawBVH) {
            throw new Error("PTRenderer: debug.drawBVH flag is not set");
        }

        this.drawBVHOptions!.setUint32(20, showParents ? 1 : 0, true);

        this.webGPUContext.device.queue.writeBuffer(
            this.drawBVHOptionsBuffer!,
            20,
            this.drawBVHOptions!,
            20,
            4,
        );
    }

    debugDrawBVH() {
        if (!this.webGPUContext.device) {
            throw new Error("PTRenderer: not initialized");
        }

        if (!this.options.debug?.drawBVH) {
            throw new Error("PTRenderer: debug.drawBVH flag is not set");
        }

        this.webGPUContext.device.queue.writeBuffer(
            this.drawBVHViewMatrixBuffer!,
            0,
            new Float32Array(this.camera!.matrixWorldInverse.elements),
        );

        this.webGPUContext.device.queue.writeBuffer(
            this.drawBVHProjectionMatrixBuffer!,
            0,
            new Float32Array(this.camera!.projectionMatrix.elements),
        );

        const commandEncoder = this.webGPUContext.device.createCommandEncoder();

        const passEncoder = commandEncoder.beginRenderPass({
            colorAttachments: [
                {
                    view: this.canvasContext!.getCurrentTexture().createView(),
                    loadOp: "load",
                    storeOp: "store",
                },
            ],
        });

        const nodesLength = this.sceneBVHNodesBuffer!.size / 20;

        this.webGPUBVH!.encodeDrawBVHPass(
            passEncoder,
            nodesLength,
            this.drawBVHViewMatrixBuffer!,
            this.drawBVHProjectionMatrixBuffer!,
            this.depthTextureSampler!,
            this.depthTexture!.createView(),
            this.drawBVHOptionsBuffer!,
        );

        passEncoder.end();

        const gpuCommands = commandEncoder.finish();
        this.webGPUContext.device.queue.submit([gpuCommands]);
    }

    readjust() {
        if (!this.webGPUContext.device) {
            throw new Error("PTRenderer: not initialized");
        }

        this.pathTraceData[0] = 0;

        const canvasTexture = this.canvasContext!.getCurrentTexture();

        this.cumulativeBuffer = this.webGPUResourceGroup!.createBuffer({
            size: canvasTexture.width * canvasTexture.height * 16,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        if (this.options.debug?.drawBVH) {
            this.depthTexture!.destroy();

            this.depthTexture = this.webGPUResourceGroup!.createTexture({
                format: "r32float",
                usage:
                    GPUTextureUsage.STORAGE_BINDING |
                    GPUTextureUsage.TEXTURE_BINDING,
                size: {
                    width: this.domElement.width,
                    height: this.domElement.height,
                },
            });
        }
    }

    dispose() {
        if (this.webGPUResourceGroup) {
            this.webGPUResourceGroup.release();
        }
    }
}

export default PTRenderer;
