/*
 *  compute shaders for constructing a binary radix tree LBVH
 *  https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/
 */

import { aabbShader, bvhNodeShader } from "./structs";

export const computeRootBoundsShader = (
    workgroupSize: number,
    reduceSize: number,
) => `
    ${aabbShader}

    @group(0) @binding(0) var<storage, read> positions: array<array<f32, 3>>;
    @group(0) @binding(1) var<storage, read_write> output_aabbs: array<AABB>;

    @compute @workgroup_size(${workgroupSize}, 3)
    fn main(
        @builtin(global_invocation_id) global_id: vec3u,
        @builtin(local_invocation_id) local_id: vec3u
    ) {
        let gid = global_id.x;
        let axis = local_id.y;
        let g_offset = ${reduceSize} * gid;
        let positions_length = arrayLength(&positions);
        let output_aabbs_length = arrayLength(&output_aabbs);

        if (gid >= output_aabbs_length) {
            return;
        }

        var pi_axis = positions[g_offset][axis];
        var axis_min = pi_axis;
        var axis_max = pi_axis;

        for (var i = g_offset + 1; i < min(positions_length, g_offset + ${reduceSize}); i++) {
            pi_axis = positions[i][axis];
            axis_min = min(axis_min, pi_axis);
            axis_max = max(axis_max, pi_axis);
        }

        output_aabbs[gid].min[axis] = axis_min;
        output_aabbs[gid].max[axis] = axis_max;
    }
`;

export const computeParentBoundsShader = (
    workgroupSizeX: number,
    reduceSize: number,
) => `
    ${aabbShader}

    @group(0) @binding(0) var<storage, read> input_aabbs: array<AABB>;
    @group(0) @binding(1) var<storage, read_write> output_aabbs: array<AABB>;

    @compute @workgroup_size(${workgroupSizeX}, 3)
    fn main(
        @builtin(global_invocation_id) global_id: vec3u,
        @builtin(local_invocation_id) local_id: vec3u
    ) {
        let gid = global_id.x;
        let axis = local_id.y;
        let g_offset = ${reduceSize} * gid;
        let input_aabbs_length = arrayLength(&input_aabbs);
        let output_aabbs_length = arrayLength(&output_aabbs);

        if (gid >= output_aabbs_length) {
            return;
        }

        var axis_min = input_aabbs[g_offset].min[axis];
        var axis_max = input_aabbs[g_offset].max[axis];

        for (var i = g_offset + 1; i < min(input_aabbs_length, g_offset + ${reduceSize}); i++) {
            axis_min = min(axis_min, input_aabbs[i].min[axis]);
            axis_max = max(axis_max, input_aabbs[i].max[axis]);
        }

        output_aabbs[gid].min[axis] = axis_min;
        output_aabbs[gid].max[axis] = axis_max;
    }
`;

export const computeMortonCodesShader = (workgroupSize: number) => `
    ${aabbShader}

    @group(0) @binding(0) var<storage, read> positions: array<f32>;
    @group(0) @binding(1) var<storage, read> index: array<u32>;
    @group(0) @binding(2) var<storage, read> aabb: AABB;
    @group(0) @binding(3) var<storage, read_write> morton_codes: array<u32>;

    fn expandBits(v: u32) -> u32 {
        var _v = v;
        _v = (_v * 0x00010001u) & 0xFF0000FFu;
        _v = (_v * 0x00000101u) & 0x0F00F00Fu;
        _v = (_v * 0x00000011u) & 0xC30C30C3u;
        _v = (_v * 0x00000005u) & 0x49249249u;
        return _v;
    }
    
    fn morton3D(v: vec3f) -> u32 {
        let x = min(max(v.x * 1024.0f, 0.0f), 1023.0f);
        let y = min(max(v.y * 1024.0f, 0.0f), 1023.0f);
        let z = min(max(v.z * 1024.0f, 0.0f), 1023.0f);
        let xx = expandBits(u32(x));
        let yy = expandBits(u32(y));
        let zz = expandBits(u32(z));
        return (xx << 2) | (yy << 1) | zz;
    }

    @compute @workgroup_size(${workgroupSize})
    fn main(
        @builtin(global_invocation_id) global_id: vec3u
    ) {
        let g_id = global_id.x;
        let triangles_length = arrayLength(&morton_codes);

        if (g_id >= triangles_length) {
            return;
        }

        let tri_a = index[3 * g_id + 0];
        let tri_b = index[3 * g_id + 1];
        let tri_c = index[3 * g_id + 2];

        let tri_av = vec3f(positions[3 * tri_a + 0], positions[3 * tri_a + 1], positions[3 * tri_a + 2]);
        let tri_bv = vec3f(positions[3 * tri_b + 0], positions[3 * tri_b + 1], positions[3 * tri_b + 2]);
        let tri_cv = vec3f(positions[3 * tri_c + 0], positions[3 * tri_c + 1], positions[3 * tri_c + 2]);

        let centroid = (tri_av + tri_bv + tri_cv) / 3.0;
        let centroid_relative = (centroid - aabb.min) / (aabb.max - aabb.min);

        morton_codes[g_id] = morton3D(centroid_relative);
    }
`;

export const computeBVHShader = (workgroupSize: number) => `
    ${bvhNodeShader}

    @group(0) @binding(0) var<storage, read> sorted_morton_codes: array<u32>;
    @group(0) @binding(1) var<storage, read> sorted_ids: array<u32>;
    @group(0) @binding(2) var<storage, read_write> nodes: array<Node>;

    fn determineRange(split: u32) -> vec2u {
        let sorted_morton_codes_length = arrayLength(&sorted_morton_codes);

        if (split == 0) {
            return vec2(0, sorted_morton_codes_length - 1);
        }

        let split_val = sorted_morton_codes[split];
        let prefix_length_next = countLeadingZeros(split_val ^ sorted_morton_codes[split + 1]);
        let prefix_length_prev = countLeadingZeros(split_val ^ sorted_morton_codes[split - 1]);
        let dir = sign(i32(prefix_length_next) - i32(prefix_length_prev));
        var end1 = split;
        var end2 = split;

        if (dir == 0) {
            var left_step = i32(end1);

            while (left_step > 1) {
                left_step = (left_step + 1) >> 1;
                let new_end1 = i32(end1) - left_step;

                if (new_end1 >= 0 && sorted_morton_codes[new_end1] == split_val) {
                    end1 = u32(new_end1);
                }
            }

            var right_step = i32(sorted_morton_codes_length - end2);

            while (right_step > 1) {
                right_step = (right_step + 1) >> 1;
                let new_end2 = end2 + u32(right_step);

                if (new_end2 < sorted_morton_codes_length && sorted_morton_codes[new_end2] == split_val) {
                    end2 = new_end2;
                }
            }

            while (end1 < end2) {
                let mid = end1 + ((end2 - end1) >> 1);

                if (split == mid) {
                    return vec2(end1, split);
                }
                else if (split == mid + 1) {
                    return vec2(split, end2);
                }
                
                if (split < mid) {
                    end2 = mid;
                }
                else {
                    end1 = mid + 1;
                }
            }

            return vec2(end1, end2);
        }
        else {
            let split_next = i32(split) - dir;
            let common_prefix_length = countLeadingZeros(split_val ^ sorted_morton_codes[split_next]);

            var step = i32(select(end2, sorted_morton_codes_length - end2, dir == 1));
    
            while (step > 1) {
                step = (step + 1) >> 1;
                let new_end2 = i32(end2) + dir * step;
                
                if (new_end2 >= 0 && u32(new_end2) < sorted_morton_codes_length) {
                    let new_common_prefix_length = countLeadingZeros(split_val ^ sorted_morton_codes[new_end2]);
    
                    if (new_common_prefix_length > common_prefix_length) {
                        end2 = u32(new_end2);
                    }
                }
            }
         
            return vec2(min(end1, end2), max(end1, end2));
        }
    }

    fn findSplit(first: u32, last: u32) -> u32 {
        let sorted_morton_codes_length = arrayLength(&sorted_morton_codes);
        let first_val = sorted_morton_codes[first];
        let last_val = sorted_morton_codes[last];

        if (first_val == last_val) {
            return first + ((last - first) >> 1);
        }

        let common_prefix_length = countLeadingZeros(first_val ^ last_val);
        
        var start = first;
        var end = last;
        var split = start;

        while (start <= end) {
            let new_split = start + ((end - start) >> 1);
            let new_split_val = sorted_morton_codes[new_split];
            let new_common_prefix_length = countLeadingZeros(first_val ^ new_split_val);

            if (new_common_prefix_length > common_prefix_length) {
                split = new_split;
                start = new_split + 1;
            }

            else {
                end = new_split - 1;
            }
        }

        return split;
    }

    @compute @workgroup_size(${workgroupSize})
    fn main(
        @builtin(global_invocation_id) global_id: vec3u
    ) {
        let g_id = global_id.x;
        let nodes_length = arrayLength(&nodes);

        if (g_id >= nodes_length) {
            return;
        }

        let range = determineRange(g_id);
        let first = range.x;
        let last = range.y;

        let split = findSplit(first, last);
        let left = split;
        let right = split + 1;
        let left_is_leaf = select(0u, 1u, first == left);
        let right_is_leaf = select(0u, 1u, last == right);

        nodes[g_id].left_is_leaf = left_is_leaf;
        nodes[g_id].right_is_leaf = right_is_leaf;
        nodes[g_id].left = select(left, sorted_ids[left], left_is_leaf == 1);
        nodes[g_id].right = select(right, sorted_ids[right], right_is_leaf == 1);

        if (left_is_leaf == 0) {
            nodes[left].parent = g_id;
        }

        if (right_is_leaf == 0) {
            nodes[right].parent = g_id;
        }
    }
`;

export const computeBVHBoundsShader = (workgroupSize: number) => `
    ${bvhNodeShader}
    ${aabbShader}

    @group(0) @binding(0) var<storage, read> positions: array<f32>;
    @group(0) @binding(1) var<storage, read> index: array<u32>;
    @group(0) @binding(2) var<storage, read> nodes: array<Node>;
    @group(0) @binding(3) var<storage, read_write> node_bounds: array<AABB>;
    @group(0) @binding(4) var<storage, read_write> node_atomics: array<atomic<u32>>;

    @compute @workgroup_size(${workgroupSize})
    fn main(
        @builtin(global_invocation_id) global_id: vec3u
    ) {
        let g_id = global_id.x;
        let node_bounds_length = arrayLength(&node_bounds);

        if (g_id >= node_bounds_length) {
            return;
        }

        var node_index = g_id;
        var node = nodes[node_index];

        if (node.left_is_leaf == 0 || node.right_is_leaf == 0) {
            return;
        }

        loop {
            if (node.left_is_leaf == 0 && node.right_is_leaf == 0) {
                if (atomicAdd(&node_atomics[node_index], 1) == 0) {
                    return;
                }
            }

            var left_aabb: AABB;
            var right_aabb: AABB;

            if (node.left_is_leaf == 1) {
                let tri_a = index[3 * node.left + 0];
                let tri_b = index[3 * node.left + 1];
                let tri_c = index[3 * node.left + 2];

                let tri_av = vec3f(positions[3 * tri_a + 0], positions[3 * tri_a + 1], positions[3 * tri_a + 2]);
                let tri_bv = vec3f(positions[3 * tri_b + 0], positions[3 * tri_b + 1], positions[3 * tri_b + 2]);
                let tri_cv = vec3f(positions[3 * tri_c + 0], positions[3 * tri_c + 1], positions[3 * tri_c + 2]);

                left_aabb.min = min(min(tri_av, tri_bv), tri_cv);
                left_aabb.max = max(max(tri_av, tri_bv), tri_cv);
            }

            else {
                left_aabb = node_bounds[node.left];
            }

            if (node.right_is_leaf == 1) {
                let tri_a = index[3 * node.right + 0];
                let tri_b = index[3 * node.right + 1];
                let tri_c = index[3 * node.right + 2];

                let tri_av = vec3f(positions[3 * tri_a + 0], positions[3 * tri_a + 1], positions[3 * tri_a + 2]);
                let tri_bv = vec3f(positions[3 * tri_b + 0], positions[3 * tri_b + 1], positions[3 * tri_b + 2]);
                let tri_cv = vec3f(positions[3 * tri_c + 0], positions[3 * tri_c + 1], positions[3 * tri_c + 2]);

                right_aabb.min = min(min(tri_av, tri_bv), tri_cv);
                right_aabb.max = max(max(tri_av, tri_bv), tri_cv);
            }

            else {
                right_aabb = node_bounds[node.right];
            }

            node_bounds[node_index] = AABB(
                min(left_aabb.min, right_aabb.min), 
                max(left_aabb.max, right_aabb.max)
            );

            if (node_index == 0) {
                return;
            }

            node_index = node.parent;
            node = nodes[node_index];
        }
    }
`;
