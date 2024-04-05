import { aabbShader, bvhNodeShader } from "./structs";

export const nodesToLinesShader = (workgroupSize: number) => `
    ${bvhNodeShader}
    ${aabbShader}

    @group(0) @binding(0) var<storage, read> nodes: array<Node>;
    @group(0) @binding(1) var<storage, read> node_bounds: array<AABB>;
    @group(0) @binding(2) var<storage, read_write> vertex: array<vec4f>;
    @group(0) @binding(3) var<storage, read_write> levels: array<u32>;
    
    fn swap(v1: ptr<function, vec4f>, v2: ptr<function, vec4f>, axis: u32) {
        let temp = (*v1)[axis];
        (*v1)[axis] = (*v2)[axis];
        (*v2)[axis] = temp;
    }

    fn write_lines(aabb_min: ptr<function, vec4f>, aabb_max: ptr<function, vec4f>, current_i: ptr<function, u32>) {
        for (var axis = 0u; axis < 3; axis++) {
            vertex[*current_i] = *aabb_min;
            (*current_i)++;
            swap(aabb_min, aabb_max, axis);
        
            vertex[*current_i] = *aabb_min;
            (*current_i)++;
            swap(aabb_min, aabb_max, axis);
        }
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

        var level = 0u;
        var node = g_id;

        while (node != 0) {
            node = nodes[node].parent;
            level++;
        }

        var aabb_min = vec4(node_bounds[g_id].min, 1);
        var aabb_max = vec4(node_bounds[g_id].max, 1);
        var current_i = 24 * g_id;

        write_lines(&aabb_min, &aabb_max, &current_i);

        swap(&aabb_min, &aabb_max, 0);
        swap(&aabb_min, &aabb_max, 2);
        write_lines(&aabb_min, &aabb_max, &current_i);
        
        swap(&aabb_min, &aabb_max, 1);
        swap(&aabb_min, &aabb_max, 2);
        write_lines(&aabb_min, &aabb_max, &current_i);
        
        swap(&aabb_min, &aabb_max, 0);
        swap(&aabb_min, &aabb_max, 2);
        write_lines(&aabb_min, &aabb_max, &current_i);
        
        for (var i = 0u; i < 24; i++) {
            levels[24 * g_id + i] = level;
        }
    }
`;

export const drawBVHVertexShader = `
    @group(0) @binding(0) var<uniform> view_matrix: mat4x4f;
    @group(0) @binding(1) var<uniform> projection_matrix: mat4x4f;

    struct VertexOutput {
        @builtin(position) position: vec4f,
        @location(0) v_level: f32,
        @location(1) v_depth: f32
    }

    @vertex
    fn main(
        @location(0) position: vec4f, 
        @location(1) level: u32
    ) -> VertexOutput {
        let view_position = view_matrix * position;
        let projected_position = projection_matrix * view_position;
        return VertexOutput(projected_position, f32(level), -view_position.z);
    }
`;

export const drawBVHFragmentShader = `
    struct Options {
        color: vec4f,
        level: u32,
        show_parents: u32,
    }

    @group(0) @binding(2) var depth_sampler: sampler;
    @group(0) @binding(3) var depth_texture: texture_2d<f32>;
    @group(0) @binding(4) var<uniform> options: Options;

    @fragment
    fn main(
        @builtin(position) position: vec4f, 
        @location(0) v_level: f32,
        @location(1) v_depth: f32
    ) -> @location(0) vec4f {
        if (options.show_parents == 1 && v_level >= f32(options.level) + 0.5) {
            discard;
        }

        if (
            options.show_parents == 0 && 
            (v_level < f32(options.level) - 0.5 || v_level > f32(options.level) + 0.5)
        ) {
            discard;
        }

        let depth_texture_dimensions = textureDimensions(depth_texture);
        
        var normalized_position = position;
        normalized_position.x /= f32(depth_texture_dimensions.x);
        normalized_position.y /= f32(depth_texture_dimensions.y);

        let depth = textureSample(depth_texture, depth_sampler, normalized_position.xy).r;

        if (depth < v_depth) {
            discard;
        }

        return options.color;
    }
`;
