import { conditionalCode } from "./utils";

// https://medium.com/nerd-for-tech/understanding-implementation-of-work-efficient-parallel-prefix-scan-cca2d5335c9b
export const scanShader = (workgroupSize: number, recurseNeeded: boolean) => {
    const workgroupSizeDouble = workgroupSize << 1;

    return `
        @group(0) @binding(0) var<storage, read_write> input: array<u32>;
        ${conditionalCode(recurseNeeded, "@group(0) @binding(1) var<storage, read_write> sums: array<u32>;")}

        var<workgroup> pref: array<u32, ${workgroupSizeDouble}>;

        @compute @workgroup_size(${workgroupSize})
        fn main(
            @builtin(local_invocation_id) local_id: vec3u,
            @builtin(workgroup_id) workgroup_id: vec3u
        ) {
            let n = ${workgroupSizeDouble}u;
            let lid = local_id.x;
            let wid = workgroup_id.x;
            let w_offset = wid * n;
            let gid = w_offset + lid;
            let input_length = arrayLength(&input);
            var mask = 0u;

            if (gid < input_length) {
                pref[lid] = input[gid];
            }

            if (gid + ${workgroupSize} < input_length) {
                pref[lid + ${workgroupSize}] = input[gid + ${workgroupSize}];
            }

            for (var d = 1u; d <= (n >> 1); d <<= 1) {
                workgroupBarrier();

                if ((lid & mask) == 0) {
                    let b = 2 * (lid + d) - 1;
                    let a = 2 * lid + d - 1;
                    pref[b] += pref[a];
                }

                mask = (mask << 1) | 1;
            }

            if (lid == 0) {
                ${conditionalCode(recurseNeeded, "sums[wid] = pref[n-1];")}
                pref[n-1] = 0;
            }

            for (var d = (n >> 1); d > 0; d >>= 1) {
                mask >>= 1;
                workgroupBarrier();

                if ((lid & mask) == 0) {
                    let b = 2 * (lid + d) - 1;
                    let a = 2 * lid + d - 1;

                    let t = pref[a];
                    pref[a] = pref[b];
                    pref[b] = t + pref[b];
                }
            }

            workgroupBarrier();

            if (gid < input_length) {
                input[gid] = pref[lid];
            }

            if (gid + ${workgroupSize} < input_length) {
                input[gid + ${workgroupSize}] = pref[lid + ${workgroupSize}];
            }
        }
    `;
};

export const recurseShader = (workgroupSize: number) => {
    const workgroupSizeDouble = workgroupSize << 1;

    return `
        @group(0) @binding(0) var<storage, read_write> input: array<u32>;
        @group(0) @binding(1) var<storage, read> incr: array<u32>;

        @compute @workgroup_size(${workgroupSize})
        fn main(
            @builtin(local_invocation_id) local_id: vec3u,
            @builtin(workgroup_id) workgroup_id: vec3u
        ) {
            let wid = workgroup_id.x;
            let gid = (wid + 1) * ${workgroupSizeDouble} + local_id.x;
            let input_length = arrayLength(&input);

            if (gid < input_length) {
                input[gid] += incr[wid + 1];
            }

            if (gid + ${workgroupSize} < input_length) {
                input[gid + ${workgroupSize}] += incr[wid + 1];
            }
        }
    `;
};

export const countShader = (workgroupSize: number) => `
    @group(0) @binding(0) var<storage, read> input: array<u32>;
    @group(0) @binding(1) var<storage, read_write> counts: array<atomic<u32>>;

    override iter: u32;

    @compute @workgroup_size(${workgroupSize})
    fn main(
        @builtin(global_invocation_id) global_id: vec3u,
        @builtin(workgroup_id) workgroup_id: vec3u
    ) {
        let gid = global_id.x;
        let wid = workgroup_id.x;
        let input_length = arrayLength(&input);
        let counts_length_by_16 = arrayLength(&counts) >> 4;

        if (gid < input_length) {
            let key = (input[gid] >> (iter << 2)) & 15;
            let loc = key * counts_length_by_16 + wid;

            atomicAdd(&counts[loc], 1);
        }
    }
`;

export const scatterShader = (workgroupSize: number) => `
    @group(0) @binding(0) var<storage, read> input: array<u32>;
    @group(0) @binding(1) var<storage, read> input_ids: array<u32>;
    @group(0) @binding(2) var<storage, read_write> output: array<u32>;
    @group(0) @binding(3) var<storage, read_write> output_ids: array<u32>;
    @group(0) @binding(4) var<storage, read_write> counts: array<u32>;
    
    override iter = 0u;
    var<private> lcounts: array<u32, 16>;

    @compute @workgroup_size(${workgroupSize})
    fn main(
        @builtin(global_invocation_id) global_id: vec3u
    ) {
        let gid = global_id.x;
        let start = gid * ${workgroupSize};
        let input_length = arrayLength(&input);
        let counts_length_by_16 = arrayLength(&counts) >> 4;

        if (start >= input_length) {
            return;
        }

        for (var i = 0u; i < 16; i++) {
            let ci = i * counts_length_by_16 + gid;
            lcounts[i] = counts[ci];
            counts[ci] = 0;
        }

        for (var i = start; i < min(input_length, start + ${workgroupSize}); i++) {
            let val = input[i];
            let key = (val >> (iter << 2)) & 15;
            let loc = lcounts[key];

            output[loc] = val;
            output_ids[loc] = select(input_ids[i], i, iter == 0);
            lcounts[key]++;
        }
    }
`;
