import { useRef, useEffect } from "react";
import {
    MeshStandardMaterial,
    Mesh,
    Scene,
    PerspectiveCamera,
    SphereGeometry,
    PlaneGeometry,
} from "three";
import { OrbitControls } from "three/examples/jsm/Addons.js";
import { GLTFLoader } from "three/examples/jsm/Addons.js";
import * as BufferGeometryUtils from "three/addons/utils/BufferGeometryUtils.js";
import Stats from "stats.js";
import { Pane } from "tweakpane";
import { PTRenderer } from "three-pt";
import modelPath from "../../assets/scene.gltf?url";

function testRT(canvas: HTMLCanvasElement) {
    const renderer = new PTRenderer({
        canvas,
        numBounces: 2,
        debug: {
            logs: true,
            drawBVH: true,
        },
    });

    renderer.setSize(innerWidth, innerHeight);

    const camera = new PerspectiveCamera(45, innerWidth / innerHeight);
    camera.position.z = 5;

    const controls = new OrbitControls(camera, canvas);
    controls.enableDamping = false;

    const stats = new Stats();
    stats.showPanel(0);
    document.body.appendChild(stats.dom);

    const pane = new Pane({
        title: "Debug",
    });

    const paneData = {
        log: {
            "model triangle count": 0,
            samples: 0,
            "bvh construction time": "",
            "path trace time": "",
            "max intersection checks": "",
            "average intersection checks": "",
            "total intersection checks": "",
            "max BVH traversal stack size": "",
        },
        bvh: {
            show: false,
            color: { r: 0, g: 255, b: 0, a: 0.3 },
            level: 5,
            "show parents": true,
        },
        pathTrace: {
            log: false,
        },
    };

    pane.addBinding(paneData.log, "model triangle count");
    pane.addBinding(paneData.log, "samples");
    pane.addBinding(paneData.log, "bvh construction time");
    pane.addBinding(paneData.log, "path trace time");
    pane.addBinding(paneData.log, "max intersection checks");
    pane.addBinding(paneData.log, "average intersection checks");
    pane.addBinding(paneData.log, "total intersection checks");
    pane.addBinding(paneData.log, "max BVH traversal stack size");

    const paneBVH = pane.addFolder({
        title: "BVH",
    });

    paneBVH.addBinding(paneData.bvh, "show");

    paneBVH
        .addBinding(paneData.bvh, "color")
        .on("change", ({ value }) =>
            renderer.debugSetDrawBVHColor(
                value.r / 255,
                value.g / 255,
                value.b / 255,
                value.a,
            ),
        );

    paneBVH
        .addBinding(paneData.bvh, "level", {
            min: 0,
            max: 32,
            step: 1,
        })
        .on("change", ({ value }) => renderer.debugSetDrawBVHLevel(value));

    paneBVH
        .addBinding(paneData.bvh, "show parents")
        .on("change", ({ value }) =>
            renderer.debugSetDrawBVHShowParents(value),
        );

    const panePathTrace = pane.addFolder({
        title: "Path Trace",
    });

    panePathTrace.addBinding(paneData.pathTrace, "log");

    let animateId: number;

    const begin = async () => {
        const { scene: model } = await new GLTFLoader().loadAsync(modelPath);

        const modelGeometry = BufferGeometryUtils.mergeGeometries(
            [
                (model.children[0].children[0].children[0] as Mesh).geometry,
                (model.children[0].children[0].children[1] as Mesh).geometry,
            ],
            false,
        );

        modelGeometry.translate(0, -0.15, 0);
        modelGeometry.scale(10, 10, 10);

        const modelMesh = new Mesh(
            modelGeometry,
            new MeshStandardMaterial({
                color: "white",
                roughness: 0.99,
                metalness: 0.2,
            }),
        );

        const lightGeometry = new SphereGeometry(2);
        lightGeometry.translate(0, 5.5, 0);

        const lightMesh = new Mesh(
            lightGeometry,
            new MeshStandardMaterial({
                color: "white",
                emissive: "white",
                emissiveIntensity: 4,
            }),
        );

        const planeGeometry = new PlaneGeometry(100, 100);
        planeGeometry.rotateX(-Math.PI / 2.0);
        planeGeometry.translate(0, -1, 0);

        const planeMesh = new Mesh(
            planeGeometry,
            new MeshStandardMaterial({
                color: "white",
                roughness: 0.8,
                metalness: 0.2,
            }),
        );

        const scene = new Scene();
        scene.add(modelMesh);
        scene.add(lightMesh);
        scene.add(planeMesh);

        paneData.log["model triangle count"] =
            modelMesh.geometry.getIndex()!.array.length / 3;

        pane.refresh();

        await renderer.initialize();

        let frame = 0;
        const animate = async () => {
            stats.begin();

            controls.update();
            renderer.render(scene, camera);

            if (paneData.bvh.show) {
                renderer.debugDrawBVH();
            }

            paneData.log.samples = renderer.pathTraceData[0];

            if (frame % 100 == 0 && paneData.pathTrace.log) {
                const log = await renderer.debugLogInfo();

                paneData.log["bvh construction time"] =
                    `${log.bvhConstructionTime}ms`;

                paneData.log["path trace time"] = `${log.pathTraceTime}ms`;

                paneData.log["max intersection checks"] =
                    log.maxIntersectionChecks.toString();

                paneData.log["average intersection checks"] =
                    log.averageIntersectionChecks.toString();

                paneData.log["total intersection checks"] =
                    log.totalIntersectionChecks.toString();

                paneData.log["max BVH traversal stack size"] =
                    log.maxBVHTraversalStackSize.toString();
            }

            pane.refresh();
            stats.end();
            animateId = requestAnimationFrame(animate);
        };

        animateId = requestAnimationFrame(animate);
    };

    const resize = () => {
        renderer.setSize(innerWidth, innerHeight);
        renderer.readjust();
        camera.aspect = innerWidth / innerHeight;
        camera.updateProjectionMatrix();
    };

    addEventListener("resize", resize);

    begin();

    return () => {
        renderer.dispose();
        stats.dom.remove();
        pane.dispose();
        cancelAnimationFrame(animateId);
        removeEventListener("resize", resize);
    };
}

function Canvas() {
    useEffect(() => {
        const canvas = canvasRef.current!;

        return testRT(canvas);
    }, []);

    const canvasRef = useRef<HTMLCanvasElement>(null);

    return <canvas ref={canvasRef}></canvas>;
}

export default Canvas;
