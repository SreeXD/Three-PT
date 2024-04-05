import {
    triangleShader,
    planeShader,
    rayShader,
    materialShader,
    objectShader,
    bvhNodeShader,
    aabbShader,
    perspectiveCameraShader,
    rayIntersectionShader,
    pathTraceDataShader,
    sampleDataShader,
} from "./structs";

import { conditionalCode, constants } from "./utils";

export const pathTraceShader = (
    workgroupSizeX: number,
    workgroupSizeY: number,
    numBounces: number,
    debug: boolean,
    writeDepth: boolean,
) => {
    let debugBindGroupNo = 1;
    let depthBindGroupNo = debug ? 2 : 1;

    return `
        ${triangleShader}
        ${planeShader}
        ${rayShader}
        ${materialShader}
        ${objectShader}
        ${aabbShader}
        ${bvhNodeShader}
        ${perspectiveCameraShader}
        ${rayIntersectionShader}
        ${pathTraceDataShader}
        ${sampleDataShader}

        @group(0) @binding(0) var<storage, read> objects: array<Object>;
        @group(0) @binding(1) var<storage, read> lights: array<u32>;
        @group(0) @binding(2) var<storage, read> positions: array<f32>;
        @group(0) @binding(3) var<storage, read> normals: array<f32>;
        @group(0) @binding(4) var<storage, read> index: array<u32>;
        @group(0) @binding(5) var<storage, read> triangle_to_object: array<u32>;
        @group(0) @binding(6) var<storage, read> nodes: array<Node>;
        @group(0) @binding(7) var<storage, read> node_bounds: array<AABB>;
        @group(0) @binding(8) var<uniform> camera: PerspectiveCamera;
        @group(0) @binding(9) var<uniform> data: PathTraceData;
        @group(0) @binding(10) var<storage, read_write> cumulative: array<vec3f>;
        @group(0) @binding(11) var canvas_texture: texture_storage_2d<rgba8unorm, write>;

        ${conditionalCode(
            debug,
            `@group(${debugBindGroupNo}) @binding(0) var<storage, read_write> debug: array<atomic<u32>>;`,
        )}
        
        ${conditionalCode(
            writeDepth,
            `@group(${depthBindGroupNo}) @binding(0) var depth_texture: texture_storage_2d<r32float, write>;`,
        )}

        var<private> bvh_stack: array<u32, 64>;
        var<private> bvh_stack_top: u32 = 0;
        var<private> random_seed: u32 = 0;

        ${conditionalCode(
            debug,
            `
                var<private> intersections_checks: u32 = 0;
                var<private> max_bvh_stack_traversal_size: u32 = 0;
            `,
        )}

        // https://gist.github.com/munrocket/236ed5ba7e409b8bdf1ff6eca5dcdc39
        fn xxhash32(n: u32) -> u32 {
            var h32 = n + 374761393u;
            h32 = 668265263u * ((h32 << 17) | (h32 >> (32 - 17)));
            h32 = 2246822519u * (h32 ^ (h32 >> 15));
            h32 = 3266489917u * (h32 ^ (h32 >> 13));
            return h32^(h32 >> 16);
        }

        fn random() -> f32 {
            random_seed++; 
            return f32(xxhash32(random_seed)) / f32(0xffffffff); 
        }

        fn getTriangle(triangle_index: u32) -> Triangle {
            let a = index[3 * triangle_index + 0];
            let b = index[3 * triangle_index + 1];
            let c = index[3 * triangle_index + 2];

            var triangle = Triangle(
                vec3f(positions[3 * a + 0], positions[3 * a + 1], positions[3 * a + 2]),
                vec3f(positions[3 * b + 0], positions[3 * b + 1], positions[3 * b + 2]),
                vec3f(positions[3 * c + 0], positions[3 * c + 1], positions[3 * c + 2]),
                vec3f(normals[3 * a + 0], normals[3 * a + 1], normals[3 * a + 2]),
                vec3f(normals[3 * b + 0], normals[3 * b + 1], normals[3 * b + 2]),
                vec3f(normals[3 * c + 0], normals[3 * c + 1], normals[3 * c + 2]),
            );

            return triangle;
        }

        fn getObjectOfTriangle(triangle_index: u32) -> Object {
            return objects[triangle_to_object[triangle_index]];
        }

        fn createRay(origin: vec3f, direction: vec3f) -> Ray {
            return Ray(origin + ${constants.epsilon} * direction, direction, 1 / direction);
        }

        // https://gamedev.stackexchange.com/questions/23743/whats-the-most-efficient-way-to-find-barycentric-coordinates
        fn triangleBarycentric(triangle: Triangle, point: vec3f) -> vec3f {
            let v0 = triangle.b - triangle.a; 
            let v1 = triangle.c - triangle.a; 
            let v2 = point - triangle.a;
            let d00 = dot(v0, v0);
            let d01 = dot(v0, v1);
            let d11 = dot(v1, v1);
            let d20 = dot(v2, v0);
            let d21 = dot(v2, v1);
            let den = d00 * d11 - d01 * d01;
            let v = (d11 * d20 - d01 * d21) / den;
            let w = (d00 * d21 - d01 * d20) / den;
            let u = 1 - v - w;
            
            return vec3f(u, v, w);
        }
        
        fn rayPlaneIntersection(ray: Ray, plane: Plane) -> f32 {
            let num = plane.distance_from_origin - dot(ray.origin, plane.normal);
            let den = dot(ray.direction, plane.normal);
            
            if (den == 0) {
                return -1;
            }

            let t = num / den;
            
            if (t < 0) {
                return -1;
            }
            
            return t;
        }
        
        fn rayTriangleIntersection(ray: Ray, triangle_index: u32, closest_intersection: ptr<function, RayIntersection>) -> RayIntersection {
            let triangle = getTriangle(triangle_index);
            let triangle_normal = normalize(cross(triangle.b - triangle.a, triangle.c - triangle.a));
            let triangle_plane_distance = dot(triangle.a, triangle_normal);
            let triangle_plane = Plane(triangle_plane_distance, triangle_normal);            
            let triangle_plane_intersection_distance = rayPlaneIntersection(ray, triangle_plane);

            if (
                triangle_plane_intersection_distance == -1 ||
                ((*closest_intersection).distance != -1 && (*closest_intersection).distance < triangle_plane_intersection_distance)
            ) {
                return RayIntersection(vec3(0), vec3(0), -1, 0);
            }
            
            let triangle_intersection_point = ray.origin + triangle_plane_intersection_distance * ray.direction;
            let triangle_barycentric = triangleBarycentric(triangle, triangle_intersection_point);
            
            if (
                triangle_barycentric.x < 0 || triangle_barycentric.x > 1 ||
                triangle_barycentric.y < 0 || triangle_barycentric.y > 1 ||
                triangle_barycentric.z < 0 || triangle_barycentric.z > 1
            ) {
                return RayIntersection(vec3(0), vec3(0), -1, 0);
            }
            
            let triangle_intersection_normal = normalize(triangle_barycentric.x * triangle.normal_a + triangle_barycentric.y * triangle.normal_b + triangle_barycentric.z * triangle.normal_c);
            return RayIntersection(triangle_intersection_point, triangle_intersection_normal, triangle_plane_intersection_distance, triangle_index);
        }

        // https://tavianator.com/2011/ray_box.html
        fn rayAABBIntersection(ray: Ray, aabb: AABB) -> f32 {
            var tmin = ${-constants.inf};
            var tmax = ${constants.inf};
            let t1 = (aabb.min - ray.origin) * ray.direction_inverse;
            let t2 = (aabb.max - ray.origin) * ray.direction_inverse;
    
            tmin = max(tmin, min(t1.x, t2.x));
            tmin = max(tmin, min(t1.y, t2.y));
            tmin = max(tmin, min(t1.z, t2.z));
    
            tmax = min(tmax, max(t1.x, t2.x));
            tmax = min(tmax, max(t1.y, t2.y));
            tmax = min(tmax, max(t1.z, t2.z));

            if (tmin > tmax) {
                return -1;
            }
    
            return tmin;
        }
    
        fn rayBVHIntersection(ray: Ray) -> RayIntersection {
            var intersection = RayIntersection(vec3f(0), vec3f(0), -1, 0);
            
            bvh_stack[bvh_stack_top] = 0;
            bvh_stack_top++;
    
            while(bvh_stack_top > 0) {
                ${conditionalCode(debug, "max_bvh_stack_traversal_size = max(max_bvh_stack_traversal_size, bvh_stack_top);")}

                bvh_stack_top--;
                let node_index = bvh_stack[bvh_stack_top];
    
                let aabb = node_bounds[node_index];
                let node = nodes[node_index];
    
                ${conditionalCode(debug, "intersections_checks++;")}
                
                let aabb_intersection_distance = rayAABBIntersection(ray, aabb);

                if (
                    aabb_intersection_distance != -1 && 
                    (intersection.distance == -1 || aabb_intersection_distance < intersection.distance)
                ) {
                    if (node.left_is_leaf == 1) {
                        ${conditionalCode(debug, "intersections_checks++;")}

                        let triangle_intersection = rayTriangleIntersection(ray, node.left, &intersection);
    
                        if (
                            triangle_intersection.distance != -1 && 
                            (intersection.distance == -1 || triangle_intersection.distance < intersection.distance)
                        ) {
                            intersection = triangle_intersection;
                        }
                    }
    
                    else {
                        bvh_stack[bvh_stack_top] = node.left;
                        bvh_stack_top++;
                    }
    
                    if (node.right_is_leaf == 1) {
                        ${conditionalCode(debug, "intersections_checks++;")}

                        let triangle_intersection = rayTriangleIntersection(ray, node.right, &intersection);
    
                        if (
                            triangle_intersection.distance != -1 && 
                            (intersection.distance == -1 || triangle_intersection.distance < intersection.distance)
                        ) {
                            intersection = triangle_intersection;
                        }
                    }
    
                    else {
                        bvh_stack[bvh_stack_top] = node.right;
                        bvh_stack_top++;
                    }
                }
            }
    
            return intersection;
        }

        fn sampleBRDF(object: Object, intersection: RayIntersection, out: vec3f) -> SampleData {
            let alpha = object.material.roughness * object.material.roughness;

            let phi = random() * ${2 * constants.pi};
            let theta = acos(pow(random(), 1 / (alpha + 2)));

            let x = cos(phi) * sin(theta);
            let y = cos(theta);
            let z = sin(phi) * sin(theta);

            let normal = intersection.normal;

            let tangent = normalize(select(
                vec3f(0, -normal.z, normal.y),
                vec3f(-normal.z, 0, normal.x),
                normal.x > normal.y
            ));

            let bitangent = normalize(cross(tangent, normal));
            let value = x * tangent + y * normal + z * bitangent;

            let pdf = (alpha + 2) * pow(cos(theta), alpha + 1) / ${2 * constants.pi};

            return SampleData(value, pdf);
        }

        /*
         * Cook-Torrance BRDF
         * https://graphicscompendium.com/gamedev/15-pbr
         */
        fn brdf(object: Object, intersection: RayIntersection, in: vec3f, out: vec3f) -> vec3f {
            let alpha = object.material.roughness * object.material.roughness;
            let alpha_squared = alpha * alpha;
            let halfway_vector = normalize(out - in);

            let ndoti = max(0, dot(intersection.normal, -in));
            let ndoto = max(0, dot(intersection.normal, out));
            let hdotn = max(0, dot(halfway_vector, intersection.normal));
            let odoth = max(0, dot(out, halfway_vector));

            let normal_distribution = (1 / (${constants.pi} * alpha_squared)) * pow(hdotn, 2 / alpha_squared - 2);

            let k = alpha / 2.0;
            let geometric_attenuation = max(${constants.epsilon}, ndoto) / (ndoto * (1 - k) + k);

            let fresnel = object.material.color + (1 - object.material.color) * pow(1 - odoth, 5);

            let diffuse = object.material.color * (1 - object.material.metalness) / ${constants.pi};
            let specular = normal_distribution * geometric_attenuation * fresnel / (4 * max(${constants.epsilon}, ndoti) * max(${constants.epsilon}, ndoto));

            return diffuse + specular;
        }

        fn trace(x: u32, y: u32) -> vec3f {
            let canvas_texture_dimensions = textureDimensions(canvas_texture);
            let aspect_ratio = f32(canvas_texture_dimensions.x) / f32(canvas_texture_dimensions.y);
            let n_x = ((f32(x) + 0.5) / f32(canvas_texture_dimensions.x) - 0.5) * 2.0 * aspect_ratio;
            let n_y = ((f32(y) + 0.5) / f32(canvas_texture_dimensions.y) - 0.5) * -2.0;
            let n_z = 1 / tan(radians(camera.fov) * 0.5);
            let objects_length = arrayLength(&objects);

            let ray = createRay(camera.position, normalize(n_x * camera.right + n_y * camera.up + n_z * camera.forward));
            var intersection = rayBVHIntersection(ray);

            let depth = select(${constants.inf}, dot(intersection.point - camera.position, camera.forward), intersection.distance != -1);

            ${conditionalCode(
                writeDepth,
                "textureStore(depth_texture, vec2(x, y), vec4f(depth, vec3(0)));",
            )}

            if (intersection.distance == -1 || depth < camera.near || depth > camera.far) {
                return vec3(0);
            }

            let object = getObjectOfTriangle(intersection.triangle_index);
            var color = object.material.emission.rgb * object.material.emission.a;
            var factor = vec3f(1);
            var last_intersection = intersection;
            var last_direction = ray.direction;
            var last_object = object;

            for (var i = 0; i < ${numBounces}; i++) {
                let bounce_sample = sampleBRDF(last_object, last_intersection, -last_direction);
                let bounce_direction = bounce_sample.value;
                let bounce_ray = createRay(last_intersection.point, bounce_direction);
                let bounce_intersection = rayBVHIntersection(bounce_ray);

                if (bounce_intersection.distance == -1) {
                    break;
                }

                let bounce_object = getObjectOfTriangle(bounce_intersection.triangle_index);

                factor *= brdf(last_object, last_intersection, -bounce_direction, -last_direction) * max(0, dot(last_intersection.normal, bounce_direction)) / bounce_sample.pdf;
                color += factor * bounce_object.material.emission.rgb * bounce_object.material.emission.a;

                last_intersection = bounce_intersection;
                last_direction = bounce_ray.direction;
                last_object = bounce_object;
            }

            return min(vec3f(${constants.fireflyClamp}), color);
        }

        @compute @workgroup_size(${workgroupSizeX}, ${workgroupSizeY}) 
        fn main(@builtin(global_invocation_id) global_id: vec3u) {
            let canvas_texture_dimensions = textureDimensions(canvas_texture);
            let x = global_id.x;
            let y = global_id.y;
            let pixel_id = y * canvas_texture_dimensions.x + x;
            random_seed = data.current_frame * pixel_id;

            if (x >= canvas_texture_dimensions.x || y >= canvas_texture_dimensions.y) {
                return;
            }

            let color = trace(x, y);
            let pixel_cumulative = cumulative[pixel_id] + color;
            cumulative[pixel_id] = pixel_cumulative;

            textureStore(canvas_texture, global_id.xy, vec4f(pixel_cumulative / f32(data.current_frame), 1));

            ${conditionalCode(
                debug,
                `
                    atomicMax(&debug[8], intersections_checks);
                    atomicAdd(&debug[9], intersections_checks);
                    atomicMax(&debug[10], max_bvh_stack_traversal_size);
                `,
            )}
        }
    `;
};
