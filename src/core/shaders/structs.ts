export const aabbShader = `
    struct AABB {
        min: vec3f,
        max: vec3f,
    }
`;

export const bvhNodeShader = `
    struct Node {
        left_is_leaf: u32,
        right_is_leaf: u32,
        left: u32,
        right: u32,
        parent: u32,
    }
`;

export const triangleShader = `
    struct Triangle {
        a: vec3f,
        b: vec3f,
        c: vec3f,
        normal_a: vec3f,
        normal_b: vec3f,
        normal_c: vec3f,
    }
`;

export const planeShader = `
    struct Plane {
        distance_from_origin: f32,
        normal: vec3f,
    }
`;

export const rayShader = `
    struct Ray {
        origin: vec3f,
        direction: vec3f,
        direction_inverse: vec3f,
    }
`;

export const perspectiveCameraShader = `
    struct PerspectiveCamera {
        position: vec3f,
        near: f32,
        right: vec3f,
        far: f32,
        up: vec3f,
        fov: f32,
        forward: vec3f,
    }
`;

export const materialShader = `
    struct Material {
        color: vec3f,
        emission: vec4f,
        roughness: f32,
        metalness: f32,
    }
`;

export const objectShader = `
    struct Object {
        start: u32,
        length: u32,
        material: Material,
    }
`;

export const rayIntersectionShader = `
    struct RayIntersection {
        point: vec3f,
        normal: vec3f,
        distance: f32,
        triangle_index: u32,
    }
`;

export const pathTraceDataShader = `
    struct PathTraceData {
        current_frame: u32,
    }
`;

export const sampleDataShader = `
    struct SampleData {
        value: vec3f,
        pdf: f32
    }
`;
