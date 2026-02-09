import { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import { Mesh } from 'three';

interface GeometryMorphProps {
  position?: [number, number, number];
  color?: string;
}

const GeometryMorph = ({ position = [0, 0, 0], color = "#8b5cf6" }: GeometryMorphProps) => {
  const meshRef = useRef<Mesh>(null);

  useFrame((state) => {
    if (meshRef.current) {
      // Morph between sphere and box
      const time = state.clock.elapsedTime;
      const morph = (Math.sin(time * 0.5) + 1) / 2; // 0 to 1
      
      meshRef.current.rotation.x = time * 0.3;
      meshRef.current.rotation.y = time * 0.2;
      
      // Scale effect based on morph
      meshRef.current.scale.setScalar(1 + morph * 0.3);
      
      // Color transition effect
      const intensity = 0.5 + morph * 0.5;
      (meshRef.current.material as any).emissiveIntensity = intensity;
    }
  });

  return (
    <mesh ref={meshRef} position={position}>
      <dodecahedronGeometry args={[1, 0]} />
      <meshStandardMaterial 
        color={color}
        emissive={color}
        emissiveIntensity={0.2}
        roughness={0.1}
        metalness={0.8}
      />
    </mesh>
  );
};

export default GeometryMorph;