# Three-PT

---

(This is WIP and not even close to completion)

WebGPU Path Tracer for Three.js

### Features (as of now)

-   Uses Three.js constructs for meshes, materials etc
-   Dedicated renderer for path tracing using WebGPU compute shaders
-   Accelerated path tracing using linear BVH built entirely on the GPU
-   Physically based rendering using microfacet BRDF
-   Importance sampling

### To Do

-   A Lot :l

### Usage

```js
const renderer = new PTRenderer({
    canvas,
    numBounces: 3,
});
renderer.setSize(innerWidth, innerHeight);

const camera = new THREE.PerspectiveCamera(45, innerWidth / innerHeight);
camera.position.set(0, 2, 5);
camera.lookAt(0, 0, 0);

const scene = new THREE.Scene();
scene.add(
    new THREE.Mesh(
        new THREE.SphereGeometry(1, 64, 32),
        new THREE.MeshStandardMaterial({
            color: "#bcbaff",
            roughness: 0.99,
            metalness: 0.01,
        }),
    ),
);

scene.add(
    new THREE.Mesh(
        new THREE.SphereGeometry(2).translate(0, 5.5, 0),
        new THREE.MeshStandardMaterial({
            color: "white",
            emissive: "white",
            emissiveIntensity: 4,
        }),
    ),
);

scene.add(
    new THREE.Mesh(
        new THREE.PlaneGeometry(100, 100)
            .rotateX(-Math.PI / 2.0)
            .translate(0, -1, 0),
        new THREE.MeshStandardMaterial({
            color: "white",
            roughness: 0.8,
            metalness: 0.2,
        }),
    ),
);

await renderer.initialize();

const onFrame = () => {
    renderer.render(scene, camera);
    requestAnimationFrame(onFrame);
};

requestAnimationFrame(onFrame);
```

![code snippet result](https://i.imgur.com/SKoRC9u.png)

this is how the BVH looks for the above scene
![bvh](https://i.imgur.com/HZqhD8b.png)

### Renders

![render 1](https://i.imgur.com/be5sHDE.png)
![render 2](https://imgur.com/dWL0qQa.png)
