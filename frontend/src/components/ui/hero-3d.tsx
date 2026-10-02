"use client";

import { Suspense, useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { Float, MeshDistortMaterial, Sphere, Points, PointMaterial } from "@react-three/drei";
import * as THREE from "three";

function DistortedCore() {
  const meshRef = useRef<THREE.Mesh>(null);
  useFrame((_, delta) => {
    if (meshRef.current) meshRef.current.rotation.y += delta * 0.15;
  });
  return (
    <Float speed={1.4} rotationIntensity={0.6} floatIntensity={1.2}>
      <Sphere ref={meshRef} args={[1.4, 128, 128]}>
        <MeshDistortMaterial
          color="#7c6cff"
          attach="material"
          distort={0.42}
          speed={1.8}
          roughness={0.15}
          metalness={0.6}
        />
      </Sphere>
    </Float>
  );
}

function generateStarPositions(): Float32Array {
  const arr = new Float32Array(1200 * 3);
  for (let i = 0; i < arr.length; i++) arr[i] = (Math.random() - 0.5) * 12;
  return arr;
}

const STAR_POSITIONS = generateStarPositions();

function StarField() {
  const ref = useRef<THREE.Points>(null);

  useFrame((_, delta) => {
    if (ref.current) ref.current.rotation.y += delta * 0.02;
  });

  return (
    <Points ref={ref} positions={STAR_POSITIONS} stride={3} frustumCulled>
      <PointMaterial transparent color="#4f9bff" size={0.012} sizeAttenuation depthWrite={false} />
    </Points>
  );
}

export function Hero3D({ className }: { className?: string }) {
  return (
    <div className={className}>
      <Canvas camera={{ position: [0, 0, 5], fov: 45 }} dpr={[1, 1.5]}>
        <ambientLight intensity={0.6} />
        <directionalLight position={[3, 3, 3]} intensity={1.2} />
        <Suspense fallback={null}>
          <DistortedCore />
          <StarField />
        </Suspense>
      </Canvas>
    </div>
  );
}
