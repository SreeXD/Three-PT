import WebGPUContext from "./WebGPUContext";

type WebGPUResource = GPUBuffer | GPUQuerySet | GPUTexture;

class WebGPUResourceGroup {
    context: WebGPUContext;
    resources: WebGPUResource[];

    constructor(context: WebGPUContext) {
        this.context = context;
        this.resources = [];
    }

    createBuffer(descriptor: GPUBufferDescriptor) {
        const buffer = this.context.device!.createBuffer(descriptor);
        this.resources.push(buffer);

        return buffer;
    }

    createQuerySet(descriptor: GPUQuerySetDescriptor) {
        const querySet = this.context.device!.createQuerySet(descriptor);
        this.resources.push(querySet);

        return querySet;
    }

    createTexture(descriptor: GPUTextureDescriptor) {
        const texture = this.context.device!.createTexture(descriptor);
        this.resources.push(texture);

        return texture;
    }

    async readBuffer(srcBuffer: GPUBuffer, dstBuffer?: GPUBuffer) {
        dstBuffer =
            dstBuffer ||
            this.createBuffer({
                size: srcBuffer.size,
                usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
            });

        const commandEncoder = this.context.device!.createCommandEncoder();
        commandEncoder.copyBufferToBuffer(
            srcBuffer,
            0,
            dstBuffer,
            0,
            srcBuffer.size,
        );

        const gpuCommands = commandEncoder.finish();
        this.context.device!.queue.submit([gpuCommands]);

        await dstBuffer.mapAsync(GPUMapMode.READ);
        const dstBufferMapped = dstBuffer.getMappedRange().slice(0);
        dstBuffer.unmap();

        return dstBufferMapped;
    }

    release() {
        for (const resource of this.resources) {
            resource.destroy();
        }
    }
}

export default WebGPUResourceGroup;
