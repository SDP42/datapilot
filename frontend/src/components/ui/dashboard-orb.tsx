"use client";

import { Suspense, useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { Float, MeshDistortMaterial, Sphere } from "@react-three/drei";
import * as THREE from "three";

function Orb() {
  const meshRef = useRef<THREE.Mesh>(null);
  useFrame((_, delta) => {
    if (meshRef.current) meshRef.current.rotation.y += delta * 0.1;
  });
  return (
    <Float speed={1.2} rotationIntensity={0.4} floatIntensity={0.8}>
      <Sphere ref={meshRef} args={[1.1, 96, 96]}>
        <MeshDistortMaterial
          color="#4f9bff"
          attach="material"
          distort={0.3}
          speed={1.4}
          roughness={0.2}
          metalness={0.5}
        />
      </Sphere>
    </Float>
  );
}

/** A lighter-weight relative of `Hero3D` — one ambient distorted sphere, no
 * starfield — sized for sitting behind a dashboard header rather than a full
 * hero section. */
export function DashboardOrb({ className }: { className?: string }) {
  return (
    <div className={className}>
      <Canvas camera={{ position: [0, 0, 4.5], fov: 45 }} dpr={[1, 1.5]}>
        <ambientLight intensity={0.7} />
        <directionalLight position={[3, 3, 3]} intensity={1} />
        <Suspense fallback={null}>
          <Orb />
        </Suspense>
      </Canvas>
    </div>
  );
}
